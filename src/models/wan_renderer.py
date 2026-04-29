from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor

from src.config import WanRendererConfig
from src.utils.compat import load_diffusers_classes, resolve_torch_dtype


class WanRenderer(nn.Module):
	"""Wan VAE and transformer wrapper conditioned by Qwen generation states.

	The renderer owns the paired Wan VAE and transformer. It handles latent
	encoding/decoding, removes the transformer's text cross-attention branch, and
	adds projected Qwen generation states to the Wan patch-token representation.

	Args:
		wan_config: Wan renderer branch configuration.
		condition_dim: Hidden size of the Qwen generation readout.
		gradient_checkpointing: Whether Wan blocks should use gradient checkpointing.
	"""

	def __init__(self, wan_config: WanRendererConfig, condition_dim: int, gradient_checkpointing: bool = True):
		super().__init__()
		AutoencoderKLWan, _, WanTransformer3DModel = load_diffusers_classes()
		transformer_load_kwargs = {
			"subfolder": "transformer",
			"local_files_only": True,
		}
		transformer_dtype = resolve_torch_dtype(wan_config.transformer_dtype)
		if transformer_dtype is not None:
			transformer_load_kwargs["torch_dtype"] = transformer_dtype
		self.transformer = WanTransformer3DModel.from_pretrained(wan_config.checkpoint_path, **transformer_load_kwargs)
		if gradient_checkpointing:
			if hasattr(self.transformer, "enable_gradient_checkpointing"):
				self.transformer.enable_gradient_checkpointing()
			else:
				self.transformer.gradient_checkpointing = True

		vae_load_kwargs = {
			"subfolder": "vae",
			"local_files_only": True,
		}
		vae_dtype = resolve_torch_dtype(wan_config.vae_dtype)
		if vae_dtype is not None:
			vae_load_kwargs["torch_dtype"] = vae_dtype
		self.vae = AutoencoderKLWan.from_pretrained(wan_config.checkpoint_path, **vae_load_kwargs)
		if wan_config.vae_enable_slicing and hasattr(self.vae, "enable_slicing"):
			self.vae.enable_slicing()
		if wan_config.vae_enable_tiling and hasattr(self.vae, "enable_tiling"):
			self.vae.enable_tiling()
		self.vae.requires_grad_(False)

		self._remove_text_conditioning_modules()
		self.inner_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
		trainable_dtype = self.transformer.patch_embedding.weight.dtype
		self.condition_proj = nn.Linear(condition_dim, self.inner_dim, dtype=trainable_dtype)
		self.null_cond = nn.Parameter(torch.zeros(1, 1, self.inner_dim, dtype=trainable_dtype))
		self._initialize_conditioning_parameters(wan_config)
		nn.init.normal_(self.null_cond, std=0.02)

		latents_mean = torch.tensor(
			self.vae.config.latents_mean,
			dtype=torch.float32,
		).view(1, self.vae.config.z_dim, 1, 1, 1)
		latents_recip_std = 1.0 / torch.tensor(
			self.vae.config.latents_std,
			dtype=torch.float32,
		).view(1, self.vae.config.z_dim, 1, 1, 1)
		self.latents_mean = nn.Buffer(latents_mean, persistent=False)
		self.latents_recip_std = nn.Buffer(latents_recip_std, persistent=False)

	def _initialize_conditioning_parameters(self, wan_config: WanRendererConfig) -> None:
		"""Initialize Qwen-to-Wan conditioning projection according to config.

		Zero initialization starts training from the original Wan transformer
		behavior. Normal initialization exposes the renderer to Qwen signals
		immediately while keeping the injected residual near zero. Default
		initialization preserves the `nn.Linear` constructor's parameters.

		Args:
			wan_config: Wan renderer configuration containing the init strategy.
		"""

		if wan_config.condition_proj_init == "zero":
			nn.init.zeros_(self.condition_proj.weight)
			nn.init.zeros_(self.condition_proj.bias)
		elif wan_config.condition_proj_init == "normal":
			if wan_config.condition_proj_init_std is None:
				raise ValueError("`wan_renderer.condition_proj_init_std` must be set when using normal init.")
			nn.init.normal_(self.condition_proj.weight, std=wan_config.condition_proj_init_std)
			nn.init.zeros_(self.condition_proj.bias)
		elif wan_config.condition_proj_init == "default":
			return
		else:
			raise ValueError(f"Unsupported condition projection init: {wan_config.condition_proj_init!r}.")

	def _remove_text_conditioning_modules(self) -> None:
		"""Delete cross-attention and text-conditioning modules that this wrapper never uses."""

		for block in self.transformer.blocks:
			if not hasattr(block, "attn2") or not hasattr(block, "norm2"):
				raise RuntimeError("The loaded Wan block no longer exposes `attn2`/`norm2`; update the pruning logic.")
			block.attn2 = None
			block.norm2 = None

		condition_embedder = self.transformer.condition_embedder
		if not hasattr(condition_embedder, "text_embedder"):
			raise RuntimeError("The loaded Wan condition embedder no longer exposes `text_embedder`; update the pruning logic.")
		condition_embedder.text_embedder = None
		if hasattr(condition_embedder, "image_embedder"):
			condition_embedder.image_embedder = None

	def encode_videos(self, videos: Tensor, sample_posterior: bool = False) -> Tensor:
		"""Encode GT videos into normalized Wan latent space.

		Args:
			videos: Input video tensor with shape `(B, 3, T, H, W)` in Wan pixel space.
			sample_posterior: Whether to sample from the VAE posterior instead of using its deterministic mode.

		Returns:
			Normalized latent tensor using the Wan transformer's dtype.
		"""

		vae_param = next(self.vae.parameters())
		videos = videos.to(device=vae_param.device, dtype=vae_param.dtype, non_blocking=True)
		with torch.no_grad():
			posterior = self.vae.encode(videos).latent_dist
			latents = posterior.sample() if sample_posterior else posterior.mode()
		latents = (latents.float() - self.latents_mean.to(latents.device)) * self.latents_recip_std.to(latents.device)
		return latents.to(self.transformer.patch_embedding.weight.dtype)

	def decode_latents(self, latents: Tensor) -> Tensor:
		"""Decode normalized Wan latents back into video space.

		Args:
			latents: Normalized latent tensor.

		Returns:
			Decoded video tensor in Wan output pixel range.
		"""

		vae_param = next(self.vae.parameters())
		latents = latents / self.latents_recip_std.to(latents.device) + self.latents_mean.to(latents.device)
		latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
		with torch.no_grad():
			return self.vae.decode(latents, return_dict=False)[0]

	def latent_patch_grids(self, latents: Tensor) -> Tensor:
		"""Compute the Wan patch-token grid for a latent tensor.

		Args:
			latents: Latent tensor with shape `(B, C, T, H, W)`.

		Returns:
			A tensor of shape `(B, 3)` containing `(t, h, w)` patch-grid sizes.
		"""

		p_t, p_h, p_w = self.transformer.config.patch_size
		grid = torch.tensor(
			[
				latents.size(2) // p_t,
				latents.size(3) // p_h,
				latents.size(4) // p_w,
			],
			device=latents.device,
			dtype=torch.long,
		)
		return grid.unsqueeze(0).expand(latents.size(0), -1)

	def _compute_time_embeddings(
		self,
		timestep: Tensor,
		dtype: torch.dtype,
		device: torch.device,
	) -> tuple[Tensor, Tensor]:
		"""Compute Wan time embeddings from scalar or batched timesteps.

		Args:
			timestep: Current diffusion timestep for each sample.
			dtype: Target dtype for the returned tensors.
			device: Target device.

		Returns:
			A tuple of:
			- `temb`, the base time embedding,
			- `timestep_proj`, the AdaLN modulation parameters reshaped as `(B, 6, D)`.
		"""

		condition_embedder = self.transformer.condition_embedder
		timestep = timestep.to(device=device)
		timestep_features = condition_embedder.timesteps_proj(timestep)
		timestep_features = timestep_features.to(device=device)
		temb = condition_embedder.time_embedder(timestep_features).to(dtype=dtype)
		timestep_proj = condition_embedder.time_proj(condition_embedder.act_fn(temb)).to(dtype=dtype)
		timestep_proj = timestep_proj.unflatten(1, (6, -1))
		return temb, timestep_proj

	def _apply_conditioning(self, hidden_states: Tensor, condition_hidden_states: Tensor, token_mask: Tensor | None) -> Tensor:
		"""Project Qwen states into Wan space and fill padded slots with a learned null token.

		Args:
			hidden_states: Wan patch-token hidden states with shape `(B, N, D)`.
			condition_hidden_states: Qwen generation hidden states aligned with Wan tokens.
			token_mask: Optional boolean mask identifying valid conditioning tokens.

		Returns:
			The condition tensor projected into Wan hidden space and aligned with `hidden_states`.
		"""

		condition_hidden_states = self.condition_proj(
			condition_hidden_states.to(device=hidden_states.device, dtype=hidden_states.dtype)
		)
		if token_mask is None:
			return condition_hidden_states

		null_cond = self.null_cond.to(device=hidden_states.device, dtype=hidden_states.dtype).expand_as(condition_hidden_states)
		return torch.where(token_mask.unsqueeze(-1), condition_hidden_states, null_cond)

	def _forward_block(
		self,
		block: nn.Module,
		hidden_states: Tensor,
		timestep_proj: Tensor,
		rotary_emb,
	) -> Tensor:
		"""Run one Wan transformer block without the removed cross-attention path.

		Args:
			block: One pretrained Wan transformer block.
			hidden_states: Patch-token hidden states.
			timestep_proj: Time-conditioning modulation tensor.
			rotary_emb: Wan rotary embeddings for the current latent grid.

		Returns:
			Updated hidden states after self-attention and FFN.
		"""

		shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
			block.scale_shift_table + timestep_proj.float()
		).chunk(6, dim=1)
		norm_hidden_states = (block.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
		attn_output = block.attn1(norm_hidden_states, None, None, rotary_emb)
		hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

		norm_hidden_states = (block.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
			hidden_states
		)
		ff_output = block.ffn(norm_hidden_states)
		hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
		return hidden_states

	def forward(
		self,
		hidden_states: Tensor,
		timestep: Tensor,
		condition_hidden_states: Tensor,
		token_mask: Tensor | None = None,
		return_dict: bool = True,
	) -> dict[str, Tensor] | Tensor:
		"""Predict the latent-space denoising output.

		Args:
			hidden_states: Noisy Wan latents with shape `(B, C, T, H, W)`.
			timestep: Current timestep per sample.
			condition_hidden_states: Qwen generation states aligned with Wan patch tokens.
			token_mask: Optional boolean mask for valid conditioning tokens.
			return_dict: Whether to return a dict with key `sample`.

		Returns:
			Either the denoised latent prediction tensor or `{"sample": tensor}`.
		"""

		_, _, num_frames, height, width = hidden_states.size()
		p_t, p_h, p_w = self.transformer.config.patch_size
		grid_t, grid_h, grid_w = num_frames // p_t, height // p_h, width // p_w
		dtype = self.transformer.patch_embedding.weight.dtype
		device = hidden_states.device

		rotary_emb = self.transformer.rope(hidden_states)
		hidden_states = self.transformer.patch_embedding(hidden_states.to(dtype=dtype))
		hidden_states = hidden_states.flatten(2).transpose(1, 2)
		condition_hidden_states = self._apply_conditioning(hidden_states, condition_hidden_states, token_mask)
		hidden_states = hidden_states + condition_hidden_states

		temb, timestep_proj = self._compute_time_embeddings(timestep, dtype=hidden_states.dtype, device=device)

		checkpoint_fn = getattr(self.transformer, "_gradient_checkpointing_func", None)
		for block in self.transformer.blocks:
			if torch.is_grad_enabled() and self.transformer.gradient_checkpointing and callable(checkpoint_fn):
				hidden_states = checkpoint_fn(
					self._forward_block,
					block, hidden_states, timestep_proj, rotary_emb,
				)
			else:
				hidden_states = self._forward_block(block, hidden_states, timestep_proj, rotary_emb)

		shift, scale = (self.transformer.scale_shift_table.to(device) + temb.unsqueeze(1)).chunk(2, dim=1)
		hidden_states = (self.transformer.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
		hidden_states = self.transformer.proj_out(hidden_states)

		# Unpatchification: reconstruct the `(B, C, T, H, W)` latents.
		output = rearrange(
			hidden_states,
			"b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)",
			t=grid_t, h=grid_h, w=grid_w, pt=p_t, ph=p_h, pw=p_w,
		)
		if not return_dict:
			return output
		return {"sample": output}
