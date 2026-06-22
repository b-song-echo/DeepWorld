from copy import deepcopy
import math
from typing import Any

from einops import rearrange
import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel
from torch import Tensor
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import GradientCheckpointingLayer

from vggt.vggt.models.vggt import VGGT
from src.config import DeepWorldQWBrainConfig, DeepWorldQWConfig, DeepWorldQWRendererConfig
from src.modules import LoraLinear, inject_lora_layers
from src.utils import resolve_torch_dtype


VIS_MODALITY = 0
GEO_MODALITY = 1
TXT_MODALITY = 2
GEN_MODALITY = 3


class LoraFfn(nn.Module):
	"""Modality-specific LoRA residuals on top of one shared frozen Qwen FFN.

	The base FFN weights are shared by reference with the frozen text expert, but
	each modality still performs its own FFN computation with its own LoRA
	residuals.
	"""

	def __init__(
		self, base_ffn: nn.Module,
		rank: int, alpha: int, dropout: float
	):
		super().__init__()
		self.act_fn = base_ffn.act_fn
		self.gate_lora_proj = LoraLinear(
			base_layer=base_ffn.gate_proj,
			rank=rank, alpha=alpha, dropout=dropout
		)
		self.up_lora_proj = LoraLinear(
			base_layer=base_ffn.up_proj,
			rank=rank, alpha=alpha, dropout=dropout
		)
		self.down_lora_proj = LoraLinear(
			base_layer=base_ffn.down_proj,
			rank=rank, alpha=alpha, dropout=dropout
		)

	def forward(self, hidden_states: Tensor) -> Tensor:
		"""Apply modality-specific LoRA residuals on top of the shared base FFN."""

		gating = self.act_fn(self.gate_lora_proj(hidden_states))
		middle = self.up_lora_proj(hidden_states) * gating
		return self.down_lora_proj(middle)


class MoFfn(nn.Module):
	"""Frozen Qwen FFN plus modality-specific routed FFN experts.

	The pretrained text FFN stays frozen. Three extra FFN experts are introduced
	for visual, geometry, and generation tokens. Each expert can either train all
	of its parameters (`full`) or reuse the shared frozen base FFN with dedicated
	LoRA residuals (`lora`).

	Args:
		txt_expert: The pretrained Qwen FFN to reuse as the frozen text expert.
		mode: Training mode for the three routed experts, either `full` or `lora`.
		lora_rank: LoRA rank used when `mode=lora`.
		lora_alpha: LoRA scaling factor used when `mode=lora`.
		lora_dropout: LoRA dropout probability used when `mode=lora`.
	"""

	def __init__(
		self, txt_expert: nn.Module, mode: str,
		lora_rank: int, lora_alpha: int, lora_dropout: float,
	):
		super().__init__()
		self.txt_expert = txt_expert
		self.txt_expert.requires_grad_(False)

		self.vis_expert = self._build_modality_expert(
			txt_expert, mode=mode,
			lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
		)
		self.geo_expert = self._build_modality_expert(
			txt_expert, mode=mode,
			lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
		)
		self.gen_expert = self._build_modality_expert(
			txt_expert, mode=mode,
			lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
		)

	def _build_modality_expert(
		self, txt_expert: nn.Module, mode: str,
		lora_rank: int, lora_alpha: int, lora_dropout: float,
	) -> nn.Module:
		"""Instantiate one modality-specific FFN expert."""

		if mode == "full":
			new_expert = deepcopy(txt_expert)
			new_expert.requires_grad_(True)
			return new_expert

		if mode == "lora":
			new_expert = LoraFfn(
				base_ffn=txt_expert,
				rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
			)
			return new_expert

		raise ValueError(f"Expected: `full`, `lora`.")

	def forward(self, hidden_states: Tensor, modality_ids: Tensor | None = None) -> Tensor:
		"""Route tokens to the correct FFN expert.

		Args:
			hidden_states: Tensor with shape `(..., hidden_size)`.
			modality_ids: Integer tensor aligned with the token dimension. If omitted,
				the frozen text expert is applied to all tokens.

		Returns:
			The FFN output with the same shape as `hidden_states`.
		"""

		if modality_ids is None:
			return self.txt_expert(hidden_states)

		flat_hidden = hidden_states.flatten(0, -2)
		flat_modality = modality_ids.flatten(0, -1)
		output = torch.empty_like(flat_hidden)

		for modality_id, expert in {
			VIS_MODALITY: self.vis_expert,
			GEO_MODALITY: self.geo_expert,
			TXT_MODALITY: self.txt_expert,
			GEN_MODALITY: self.gen_expert,
		}.items():
			mask = flat_modality == modality_id
			if mask.any():
				output[mask] = expert(flat_hidden[mask])

		return output.view_as(hidden_states)


class MoFfnQwenDecoderLayer(GradientCheckpointingLayer):
	"""Qwen decoder layer with the FFN replaced by hard-routed modality experts.

	Args:
		original_layer: Pretrained Qwen decoder layer to wrap.
	"""

	def __init__(
		self, original_layer: nn.Module,
		moffn_mode: str,
		moffn_lora_rank: int,
		moffn_lora_alpha: int,
		moffn_lora_dropout: float,
	):
		super().__init__()
		self.hidden_size = original_layer.hidden_size
		self.self_attn = original_layer.self_attn
		self.input_layernorm = original_layer.input_layernorm
		self.post_attention_layernorm = original_layer.post_attention_layernorm
		self.mlp = MoFfn(
			original_layer.mlp, mode=moffn_mode,
			lora_rank=moffn_lora_rank,
			lora_alpha=moffn_lora_alpha,
			lora_dropout=moffn_lora_dropout,
		)

	def forward(
		self, hidden_states: Tensor,
		modality_ids: Tensor | None = None,
		**kwargs,
	) -> Tensor:
		"""Run one decoder block with routed FFN execution.

		Args:
			hidden_states: Decoder hidden states.
			modality_ids: Token-level modality labels used for FFN routing.
			**kwargs: Forwarded extra attention kwargs.

		Returns:
			Updated hidden states after self-attention and routed FFN.
		"""

		residual = hidden_states
		hidden_states = self.input_layernorm(hidden_states)
		hidden_states, _ = self.self_attn(hidden_states=hidden_states, **kwargs)
		hidden_states = residual + hidden_states

		residual = hidden_states
		hidden_states = self.post_attention_layernorm(hidden_states)
		hidden_states = self.mlp(hidden_states, modality_ids=modality_ids)
		hidden_states = residual + hidden_states
		return hidden_states


class DeepWorldQWBrain(nn.Module):
	"""Build multimodal Qwen inputs and extract Wan-aligned generation states.

	The module combines:

	- frozen Qwen visual features,
	- frozen VGGT tokens projected into Qwen hidden space,
	- learned modality marker tokens between the multimodal spans,
	- prompt token embeddings,
	- learned generation probe tokens whose final hidden states condition Wan.

	Args:
		qwen_config: DeepWorldQW brain branch configuration.
		gradient_checkpointing: Whether to enable checkpointing in the Qwen text stack.
	"""

	def __init__(self, qwen_config: DeepWorldQWBrainConfig, gradient_checkpointing: bool = True):
		super().__init__()
		qwen_load_kwargs = {"local_files_only": True}
		transformer_dtype = resolve_torch_dtype(qwen_config.transformer_dtype)
		if transformer_dtype is not None:
			qwen_load_kwargs["torch_dtype"] = transformer_dtype

		# only Qwen3VLModel is needed, but directly loading checkpoint into it results in warnings, here is the workaround
		qwen = Qwen3VLForConditionalGeneration.from_pretrained(qwen_config.checkpoint_path, **qwen_load_kwargs).model
		qwen.requires_grad_(False)
		self.language_model = qwen.language_model
		self.vision_encoder = qwen.visual
		self.get_image_features = qwen.get_image_features
		self.hidden_size = qwen.config.text_config.hidden_size
		self.vis_patch_size = qwen.config.vision_config.spatial_merge_size

		replaced_layers = [MoFfnQwenDecoderLayer(
			original_layer=layer,
			moffn_mode=qwen_config.routed_ffn_mode.lower(),
			moffn_lora_rank=qwen_config.routed_ffn_lora_rank,
			moffn_lora_alpha=qwen_config.routed_ffn_lora_alpha,
			moffn_lora_dropout=qwen_config.routed_ffn_lora_dropout,
		) for layer in self.language_model.layers]
		self.language_model.layers = nn.ModuleList(replaced_layers)

		inject_lora_layers(
			self.language_model,
			target_names=qwen_config.lora_target_modules,
			rank=qwen_config.lora_rank,
			alpha=qwen_config.lora_alpha,
			dropout=qwen_config.lora_dropout,
		)

		trainable_dtype = next(self.language_model.parameters()).dtype
		self.vis_bridge = nn.Linear(
			self.hidden_size, self.hidden_size, bias=True, dtype=trainable_dtype,
		)
		self.txt_bridge = nn.Linear(
			self.hidden_size, self.hidden_size, bias=True, dtype=trainable_dtype,
		)
		nn.init.eye_(self.vis_bridge.weight)
		nn.init.zeros_(self.vis_bridge.bias)
		nn.init.eye_(self.txt_bridge.weight)
		nn.init.zeros_(self.txt_bridge.bias)

		if gradient_checkpointing:
			kwargs = {"use_reentrant": False}
			self.language_model.gradient_checkpointing_enable(kwargs)
			self.language_model.config.use_cache = False

		self.geometry_encoder = VGGT.from_pretrained(
			qwen_config.vggt_checkpoint_path, local_files_only=True,
			enable_camera=False, enable_point=False,
			enable_depth=False, enable_track=False,
		)
		vggt_dtype = resolve_torch_dtype(qwen_config.vggt_dtype)
		if vggt_dtype is not None:
			self.geometry_encoder.to(dtype=vggt_dtype)
		self.geometry_encoder.requires_grad_(False)

		self.geo_patch_size = self.geometry_encoder.aggregator.patch_size
		geo_hidden_size = self.geometry_encoder.aggregator.frame_blocks[0].norm1.weight.size(0) * 2
		self.geo_bridge = nn.Sequential(
			nn.Linear(geo_hidden_size, self.hidden_size, dtype=trainable_dtype),
			nn.GELU(),
			nn.Linear(self.hidden_size, self.hidden_size, dtype=trainable_dtype),
		)

		self.segment_tokens = nn.ParameterDict({
			"vis": nn.Parameter(torch.empty(1, self.hidden_size, dtype=trainable_dtype)),
			"geo": nn.Parameter(torch.empty(1, self.hidden_size, dtype=trainable_dtype)),
			"txt": nn.Parameter(torch.empty(1, self.hidden_size, dtype=trainable_dtype)),
			"gen": nn.Parameter(torch.empty(1, self.hidden_size, dtype=trainable_dtype)),
		})
		self.gen_slot_token = nn.Parameter(torch.empty(1, self.hidden_size, dtype=trainable_dtype))
		for token in self.segment_tokens.values():
			nn.init.normal_(token, std=0.02)
		nn.init.normal_(self.gen_slot_token, std=0.02)

	def _encode_visual(
		self, pixel_values: Tensor,
		image_grid_thw: Tensor,
	) -> tuple[list[Tensor], list[tuple[int, int, int]]]:
		"""Extract one sample's reference-image tokens with Qwen's frozen vision encoder."""

		device = next(self.parameters()).device
		vis_dtype = next(self.vision_encoder.parameters()).dtype
		with torch.no_grad():
			image_features = self.get_image_features(
				pixel_values=pixel_values.to(device, vis_dtype, non_blocking=True),
				image_grid_thw=image_grid_thw.to(device, non_blocking=True),
				return_dict=True,
			).pooler_output
		bridge_dtype = self.vis_bridge.weight.dtype
		if isinstance(image_features, tuple):
			feature_lengths = [features.size(0) for features in image_features]
			flat_features = torch.cat(image_features, dim=0).to(bridge_dtype)
			flat_features = self.vis_bridge(flat_features)
			vis_features = list(torch.split(flat_features, feature_lengths))
		else:
			image_features = image_features.to(bridge_dtype)
			image_features = self.vis_bridge(image_features)
			vis_features = [image_features] if image_features.dim() == 2 else list(image_features)

		vis_grids: list[tuple[int, int, int]] = []
		for grid in image_grid_thw:
			grid_t, grid_h, grid_w = grid.tolist()
			patch_size = self.vis_patch_size
			vis_grids.append((grid_t, grid_h // patch_size, grid_w // patch_size))
		return vis_features, vis_grids

	def _encode_geometry(
		self, geo_images: Tensor,
	) -> tuple[list[Tensor], list[tuple[int, int, int]]]:
		"""Extract one sample's reference-image geometry tokens with frozen VGGT."""

		device = next(self.parameters()).device
		vggt_dtype = next(self.geometry_encoder.aggregator.parameters()).dtype
		bridge_dtype = self.geo_bridge[0].weight.dtype
		with torch.no_grad():
			aggregated_tokens, patch_start_idx = self.geometry_encoder.aggregator(
				geo_images.unsqueeze(0).to(device, vggt_dtype, non_blocking=True)
			)
			geo_tokens = aggregated_tokens[-1][0, :, patch_start_idx:, :]
		geo_tokens = self.geo_bridge(geo_tokens.to(bridge_dtype))

		patch_size = self.geometry_encoder.aggregator.patch_size
		grid_h = geo_images.size(-2) // patch_size
		grid_w = geo_images.size(-1) // patch_size

		geo_features = list(geo_tokens)
		geo_grids = [(1, grid_h, grid_w) for _ in geo_features]
		return geo_features, geo_grids

	def _make_txt_positions(
		self, length: int, start_index: int, device: torch.device
	) -> tuple[Tensor, int]:
		"""Build text-style RoPE indices for a contiguous token span.

		Args:
			length: Number of tokens in the text span.
			start_index: Global sequence offset for the first token.
			device: Target device.

		Returns:
			A tuple of:
			- position tensor with shape `(3, length)`,
			- next free global position index.
		"""

		next_index = start_index + length
		positions = torch.arange(
			start_index, next_index,
			device=device, dtype=torch.long
		).unsqueeze(0).expand(3, -1)
		return positions, next_index

	def _make_grid_positions(
		self, grid_thw: tuple[int, int, int],
		start_index: int, device: torch.device,
	) -> tuple[Tensor, int]:
		"""Build multimodal `(t, h, w)` RoPE indices for a grid-shaped token span.

		Args:
			grid_thw: Token grid as `(t, h, w)`.
			start_index: Global offset added to all generated positions.
			device: Target device.

		Returns:
			A tuple of:
			- position tensor with shape `(3, t*h*w)`,
			- next free global position index.
		"""

		grid_t, grid_h, grid_w = grid_thw
		t_index = torch.arange(grid_t, device=device)
		t_index = t_index.view(-1, 1, 1).expand(-1, grid_h, grid_w).flatten()
		h_index = torch.arange(grid_h, device=device)
		h_index = h_index.view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
		w_index = torch.arange(grid_w, device=device)
		w_index = w_index.view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
		positions = torch.stack([t_index, h_index, w_index], dim=0) + start_index
		next_index = int(positions.max().item()) + 1
		return positions, next_index

	def _append_span(
		self,
		parts: list[Tensor],
		part_positions: list[Tensor],
		part_modalities: list[Tensor],
		embeddings: Tensor,
		positions: Tensor,
		modality_id: int,
	) -> None:
		"""Append one sequence span and its aligned metadata."""

		parts.append(embeddings)
		part_positions.append(positions)
		part_modalities.append(torch.full(
			(embeddings.size(0),),
			modality_id,
			device=embeddings.device,
			dtype=torch.long,
		))

	def _append_segment_token(
		self,
		token_name: str,
		modality_id: int,
		parts: list[Tensor],
		part_positions: list[Tensor],
		part_modalities: list[Tensor],
		cursor: int,
		device: torch.device,
		dtype: torch.dtype,
	) -> int:
		"""Append one learned modality marker token and advance the cursor."""

		segment_token = self.segment_tokens[token_name].to(device=device, dtype=dtype)
		positions, cursor = self._make_txt_positions(1, cursor, device)
		self._append_span(
			parts, part_positions, part_modalities,
			segment_token, positions, modality_id,
		)
		return cursor

	def _build_language_inputs(
		self,
		vis_features: list[Tensor],
		vis_grids: list[tuple[int, int, int]],
		geo_features: list[Tensor],
		geo_grids: list[tuple[int, int, int]],
		txt_input_ids: Tensor,
		txt_attention_mask: Tensor,
		latent_patch_grid: tuple[int, int, int],
	) -> dict[str, Tensor | int]:
		"""Assemble one direct-embedding Qwen sequence for the local sample."""

		device = next(self.parameters()).device
		txt_embeddings = self.language_model.embed_tokens(
			txt_input_ids.to(device=device, non_blocking=True)
		)
		txt_embeddings = self.txt_bridge(txt_embeddings)
		txt_attention_mask = txt_attention_mask.to(device, non_blocking=True).bool()

		parts: list[Tensor] = []
		part_positions: list[Tensor] = []
		part_modalities: list[Tensor] = []
		cursor = 0

		cursor = self._append_segment_token(
			"vis", VIS_MODALITY, parts, part_positions,
			part_modalities, cursor, device, txt_embeddings.dtype,
		)
		for features, grid in zip(vis_features, vis_grids):
			features = features.to(device)
			positions, cursor = self._make_grid_positions(grid, cursor, device)
			self._append_span(
				parts, part_positions, part_modalities,
				features, positions, VIS_MODALITY,
			)

		cursor = self._append_segment_token(
			"geo", GEO_MODALITY, parts, part_positions,
			part_modalities, cursor, device, txt_embeddings.dtype,
		)
		for features, grid in zip(geo_features, geo_grids):
			features = features.to(device)
			positions, cursor = self._make_grid_positions(grid, cursor, device)
			self._append_span(
				parts, part_positions, part_modalities,
				features, positions, GEO_MODALITY,
			)

		txt_features = txt_embeddings[txt_attention_mask]
		cursor = self._append_segment_token(
			"txt", TXT_MODALITY, parts, part_positions,
			part_modalities, cursor, device, txt_embeddings.dtype,
		)
		positions, cursor = self._make_txt_positions(txt_features.size(0), cursor, device)
		self._append_span(
			parts, part_positions, part_modalities,
			txt_features, positions, TXT_MODALITY,
		)

		cursor = self._append_segment_token(
			"gen", GEN_MODALITY, parts, part_positions,
			part_modalities, cursor, device, txt_embeddings.dtype,
		)
		gen_positions, cursor = self._make_grid_positions(latent_patch_grid, cursor, device)
		gen_length = math.prod(latent_patch_grid)
		gen_tokens = self.gen_slot_token.to(device=device, dtype=txt_embeddings.dtype).expand(gen_length, -1)
		self._append_span(
			parts, part_positions, part_modalities,
			gen_tokens, gen_positions, GEN_MODALITY,
		)

		sequence = torch.cat(parts, dim=0)
		position_ids = torch.cat(part_positions, dim=1)
		modality_ids = torch.cat(part_modalities, dim=0)
		sequence_length = sequence.size(0)
		gen_start = sequence_length - gen_length

		return {
			"inputs_embeds": sequence.unsqueeze(0),
			"attention_mask": torch.ones(1, sequence_length, device=device, dtype=torch.long),
			"position_ids": position_ids.unsqueeze(1),
			"modality_ids": modality_ids.unsqueeze(0),
			"gen_start": gen_start,
			"gen_length": gen_length,
		}

	def forward(self, batch: dict[str, Any], latent_patch_grid: tuple[int, int, int]) -> dict[str, Tensor]:
		"""Run the full Qwen/VGGT conditioning pipeline.

		Args:
			batch: Training batch produced by `DeepWorldQWBatchCollator`.
			latent_patch_grid: Desired Wan latent token grid for the local sample.

		Returns:
			A dictionary containing the final hidden states of the learned
			generation tokens.
		"""

		vis_features, vis_grids = self._encode_visual(
			batch["qwen_vis_pixel_values"],
			batch["qwen_vis_grid_thw"],
		)
		geo_features, geo_grids = self._encode_geometry(batch["geo_images"])
		language_inputs = self._build_language_inputs(
			vis_features=vis_features,
			vis_grids=vis_grids,
			geo_features=geo_features,
			geo_grids=geo_grids,
			txt_input_ids=batch["txt_input_ids"],
			txt_attention_mask=batch["txt_attention_mask"],
			latent_patch_grid=latent_patch_grid,
		)

		language_outputs = self.language_model(
			input_ids=None,
			inputs_embeds=language_inputs["inputs_embeds"],
			attention_mask=language_inputs["attention_mask"],
			position_ids=language_inputs["position_ids"],
			modality_ids=language_inputs["modality_ids"],
			use_cache=False
		)
		last_hidden_state = language_outputs.last_hidden_state
		gen_start = int(language_inputs["gen_start"])
		gen_length = int(language_inputs["gen_length"])

		return {
			"gen_hidden_states": last_hidden_state[:, gen_start:(gen_start + gen_length)],
		}


class DeepWorldQWRenderer(nn.Module):
	"""Wan VAE and transformer wrapper conditioned by Qwen generation states.

	The renderer owns the paired Wan VAE and transformer. It handles latent
	encoding/decoding and supports two conditioning modes:
	- `input_addition`: remove Wan text cross-attention and add projected Qwen states to patch tokens.
	- `cross_attention`: keep Wan cross-attention and replace T5 embeddings with projected Qwen states.

	Args:
		wan_config: Wan renderer branch configuration.
		condition_dim: Hidden size of the Qwen generation readout.
		gradient_checkpointing: Whether Wan blocks should use gradient checkpointing.
	"""

	def __init__(self, wan_config: DeepWorldQWRendererConfig, condition_dim: int, gradient_checkpointing: bool = True):
		super().__init__()
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

		self.condition_injection_mode = wan_config.condition_injection_mode
		self.inner_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
		trainable_dtype = self.transformer.patch_embedding.weight.dtype
		
		if self.condition_injection_mode == "input_addition":
			condition_projection_dim = self.inner_dim
		elif self.condition_injection_mode == "cross_attention":
			condition_projection_dim = int(self.transformer.config.text_dim)
		else:
			raise ValueError(f"Unsupported Wan conditioning mode: {self.condition_injection_mode!r}.")
		
		self.condition_proj = nn.Linear(condition_dim, condition_projection_dim, dtype=trainable_dtype)
		self.null_cond = nn.Parameter(torch.zeros(1, 1, condition_projection_dim, dtype=trainable_dtype))
		self._initialize_conditioning_parameters(wan_config)
		nn.init.normal_(self.null_cond, std=0.02)

		if self.condition_injection_mode == "input_addition":
			self._remove_text_conditioning_modules()
		elif self.condition_injection_mode == "cross_attention":
			self._freeze_transformer_except_cross_attention()
		else:
			raise ValueError(f"Unsupported Wan conditioning mode: {self.condition_injection_mode!r}.")

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

	def _initialize_conditioning_parameters(self, wan_config: DeepWorldQWRendererConfig) -> None:
		"""Initialize Qwen-to-Wan conditioning projection according to config.

		Zero initialization starts training from the original Wan transformer
		behavior. Normal initialization exposes the renderer to Qwen signals
		immediately while keeping the injected addition near zero. Default
		initialization preserves the `nn.Linear` constructor's parameters.

		Args:
			wan_config: Wan renderer configuration containing the init strategy.
		"""

		if wan_config.condition_proj_init == "zero":
			nn.init.zeros_(self.condition_proj.weight)
			nn.init.zeros_(self.condition_proj.bias)
		elif wan_config.condition_proj_init == "normal":
			if wan_config.condition_proj_init_std is None:
				raise ValueError("`renderer.condition_proj_init_std` must be set when using normal init.")
			nn.init.normal_(self.condition_proj.weight, std=wan_config.condition_proj_init_std)
			nn.init.zeros_(self.condition_proj.bias)
		elif wan_config.condition_proj_init == "default":
			return
		else:
			raise ValueError(f"Unsupported condition projection init: {wan_config.condition_proj_init!r}.")

	def _freeze_all_except_text_conditioning(self) -> None:
		"""Freeze all modules except text-conditioning."""

		self.transformer.requires_grad_(False)
		text_embedder = getattr(self.transformer.condition_embedder, "text_embedder", None)
		if not isinstance(text_embedder, nn.Module):
			raise RuntimeError("The loaded Wan condition embedder no longer exposes `text_embedder`; update freezing logic.")
		text_embedder.requires_grad_(True)
		for block in self.transformer.blocks:
			if not hasattr(block, "attn2") or not hasattr(block, "norm2"):
				raise RuntimeError("The loaded Wan block no longer exposes `attn2`/`norm2`; update freezing logic.")
			if not isinstance(block.attn2, nn.Module):
				raise RuntimeError("The loaded Wan cross-attention branch is not a module; update freezing logic.")
			block.attn2.requires_grad_(True)
			if isinstance(block.norm2, nn.Module):
				block.norm2.requires_grad_(True)

	def _remove_text_conditioning_modules(self) -> None:
		"""Delete text-conditioning modules that this wrapper never uses."""

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

	def latent_patch_grid(self, latents: Tensor) -> tuple[int, int, int]:
		"""Compute the Wan patch-token grid for one latent tensor.

		Args:
			latents: Latent tensor with shape `(1, C, T, H, W)`.

		Returns:
			A `(t, h, w)` patch-grid tuple.
		"""

		p_t, p_h, p_w = self.transformer.config.patch_size
		latent_t, latent_h, latent_w = latents.size()[-3:]
		if latent_t % p_t != 0 or latent_h % p_h != 0 or latent_w % p_w != 0:
			raise ValueError(
				"Wan latent shape must be divisible by transformer patch size; "
				f"latent={(latent_t, latent_h, latent_w)}, patch={(p_t, p_h, p_w)}."
			)
		return latent_t // p_t, latent_h // p_h, latent_w // p_w

	def _compute_time_embeddings(
		self,
		timestep: Tensor,
		dtype: torch.dtype,
		device: torch.device,
	) -> tuple[Tensor, Tensor]:
		"""Compute Wan time embeddings for the current single-sample timestep.

		Args:
			timestep: Current diffusion timestep with shape `(1, 1)`.
			dtype: Target dtype for the returned tensors.
			device: Target device.

		Returns:
			A tuple of:
			- `temb`, the base time embedding,
			- `timestep_proj`, the AdaLN modulation parameters reshaped as `(B, N, 6, D)`.
		"""

		condition_embedder = self.transformer.condition_embedder
		timestep = timestep.to(device=device)
		batch_size, timestep_seq_len = timestep.size()
		timestep_features = condition_embedder.timesteps_proj(timestep.flatten())
		timestep_features = timestep_features.unflatten(0, (batch_size, timestep_seq_len))

		time_embedder_dtype = next(condition_embedder.time_embedder.parameters()).dtype
		if timestep_features.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
			timestep_features = timestep_features.to(time_embedder_dtype)
		timestep_features = timestep_features.to(device=device)
		
		temb = condition_embedder.time_embedder(timestep_features).to(dtype=dtype)
		timestep_proj = condition_embedder.time_proj(condition_embedder.act_fn(temb)).to(dtype=dtype)
		timestep_proj = timestep_proj.unflatten(2, (6, -1))
		return temb, timestep_proj

	def _project_conditioning(
		self,
		condition_hidden_states: Tensor,
		drop_condition: Tensor | bool,
		device: torch.device,
		dtype: torch.dtype,
	) -> Tensor:
		"""Project Qwen states into the active Wan conditioning space.

		Args:
			condition_hidden_states: Qwen generation hidden states aligned with Wan tokens.
			drop_condition: Scalar flag that replaces the whole conditioning span
				with the learned null token.
			device: Target device.
			dtype: Target dtype.

		Returns:
			Projected condition tensor, optionally replaced by learned null tokens.
		"""

		proj_dtype = self.condition_proj.weight.dtype
		condition_hidden_states = self.condition_proj(
			condition_hidden_states.to(device=device, dtype=proj_dtype, non_blocking=True)
		).to(dtype=dtype)
		if isinstance(drop_condition, bool):
			if not drop_condition:
				return condition_hidden_states
			null_cond = self.null_cond.to(device=device, dtype=dtype).expand_as(condition_hidden_states)
			return null_cond

		drop_condition = drop_condition.to(device=device, dtype=torch.bool)
		null_cond = self.null_cond.to(device=device, dtype=dtype).expand_as(condition_hidden_states)
		return torch.where(drop_condition.view(1, 1, 1), null_cond, condition_hidden_states)

	def _forward_input_addition_block(
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
			block.scale_shift_table.unsqueeze(0) + timestep_proj.float()
		).chunk(6, dim=2)
		shift_msa = shift_msa.squeeze(2)
		scale_msa = scale_msa.squeeze(2)
		gate_msa = gate_msa.squeeze(2)
		c_shift_msa = c_shift_msa.squeeze(2)
		c_scale_msa = c_scale_msa.squeeze(2)
		c_gate_msa = c_gate_msa.squeeze(2)
		norm_hidden_states = (block.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
		attn_output = block.attn1(norm_hidden_states, None, None, rotary_emb)
		hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

		norm_hidden_states = (block.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
			hidden_states
		)
		ff_output = block.ffn(norm_hidden_states)
		hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
		return hidden_states

	def _normalize_timestep(
		self,
		timestep: Tensor,
		device: torch.device,
	) -> Tensor:
		"""Return a single-sample timestep in the 2D shape expected by Wan."""

		timestep = timestep.to(device=device)
		if timestep.dim() == 0:
			return timestep.view(1, 1)
		if timestep.dim() == 1:
			if timestep.numel() != 1:
				raise ValueError(f"DeepWorldQW expects one timestep, got shape {tuple(timestep.size())}.")
			return timestep.view(1, 1)
		if timestep.dim() == 2 and tuple(timestep.size()) == (1, 1):
			return timestep
		raise ValueError(f"DeepWorldQW expects one timestep, got shape {tuple(timestep.size())}.")

	def _forward_cross_attention(
		self,
		hidden_states: Tensor,
		timestep: Tensor,
		condition_hidden_states: Tensor,
		drop_condition: Tensor | bool,
		return_dict: bool,
	) -> dict[str, Tensor] | Tensor:
		"""Run Wan's native cross-attention path with Qwen states as text embeddings."""

		dtype = self.transformer.patch_embedding.weight.dtype
		device = hidden_states.device
		timestep = self._normalize_timestep(timestep, device)
		encoder_hidden_states = self._project_conditioning(
			condition_hidden_states,
			drop_condition,
			device=device,
			dtype=dtype,
		)

		output = self.transformer(
			hidden_states=hidden_states.to(dtype=dtype),
			timestep=timestep[:, 0] if timestep.size(1) == 1 else timestep,
			encoder_hidden_states=encoder_hidden_states,
			return_dict=False,
		)[0]
		if not return_dict:
			return output
		return {"sample": output}

	def _forward_input_addition(
		self,
		hidden_states: Tensor,
		timestep: Tensor,
		condition_hidden_states: Tensor,
		drop_condition: Tensor | bool,
		return_dict: bool,
	) -> dict[str, Tensor] | Tensor:
		"""Run the input-addition Qwen-to-Wan conditioning path."""

		p_t, p_h, p_w = self.transformer.config.patch_size
		grid_t, grid_h, grid_w = self.latent_patch_grid(hidden_states)
		dtype = self.transformer.patch_embedding.weight.dtype
		device = hidden_states.device

		timestep = self._normalize_timestep(timestep, device)
		rotary_emb = self.transformer.rope(hidden_states)
		hidden_states = self.transformer.patch_embedding(hidden_states.to(dtype=dtype))
		hidden_states = hidden_states.flatten(2).transpose(1, 2)
		condition_hidden_states = self._project_conditioning(
			condition_hidden_states,
			drop_condition,
			device=device,
			dtype=hidden_states.dtype,
		)
		hidden_states = hidden_states + condition_hidden_states

		temb, timestep_proj = self._compute_time_embeddings(timestep, dtype=hidden_states.dtype, device=device)

		checkpoint_fn = getattr(self.transformer, "_gradient_checkpointing_func", None)
		for block in self.transformer.blocks:
			if torch.is_grad_enabled() and self.transformer.gradient_checkpointing and callable(checkpoint_fn):
				hidden_states = checkpoint_fn(
					self._forward_input_addition_block,
					block, hidden_states, timestep_proj, rotary_emb,
				)
			else:
				hidden_states = self._forward_input_addition_block(block, hidden_states, timestep_proj, rotary_emb)

		shift, scale = (self.transformer.scale_shift_table.unsqueeze(0).to(device) + temb.unsqueeze(2)).chunk(2, dim=2)
		shift = shift.squeeze(2)
		scale = scale.squeeze(2)
		hidden_states = (self.transformer.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
		hidden_states = self.transformer.proj_out(hidden_states)

		output = rearrange(
			hidden_states,
			"b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)",
			t=grid_t, h=grid_h, w=grid_w, pt=p_t, ph=p_h, pw=p_w,
		)
		if not return_dict:
			return output
		return {"sample": output}

	def forward(
		self,
		hidden_states: Tensor,
		timestep: Tensor,
		condition_hidden_states: Tensor,
		drop_condition: Tensor | bool = False,
		return_dict: bool = True,
	) -> dict[str, Tensor] | Tensor:
		"""Predict the latent-space denoising output.

		Args:
			hidden_states: Noisy Wan latents with shape `(1, C, T, H, W)`.
			timestep: Current timestep for the single local sample.
			condition_hidden_states: Qwen generation states aligned with Wan patch tokens.
			drop_condition: Whether to replace conditioning with learned null tokens.
			return_dict: Whether to return a dict with key `sample`.

		Returns:
			Either the denoised latent prediction tensor or `{"sample": tensor}`.
		"""

		if self.condition_injection_mode == "cross_attention":
			return self._forward_cross_attention(
				hidden_states=hidden_states,
				timestep=timestep,
				condition_hidden_states=condition_hidden_states,
				drop_condition=drop_condition,
				return_dict=return_dict,
			)
		elif self.condition_injection_mode == "input_addition":
			return self._forward_input_addition(
				hidden_states=hidden_states,
				timestep=timestep,
				condition_hidden_states=condition_hidden_states,
				drop_condition=drop_condition,
				return_dict=return_dict,
			)
		else:
			raise ValueError(f"Unsupported Wan conditioning mode: {self.condition_injection_mode!r}.")


def build_wan_scheduler(config: DeepWorldQWConfig, scheduler_cls):
	"""Load the Wan scheduler while preserving checkpoint-specific settings.

	Args:
		config: Root configuration object.
		scheduler_cls: Imported diffusers scheduler class.

	Returns:
		A scheduler instance loaded from the Wan checkpoint when possible.
	"""

	load_kwargs = {
		"subfolder": "scheduler",
		"local_files_only": True,
	}
	try:
		if hasattr(scheduler_cls, "from_pretrained"):
			scheduler = scheduler_cls.from_pretrained(config.renderer.checkpoint_path, **load_kwargs)
		elif hasattr(scheduler_cls, "load_config"):
			scheduler_config = scheduler_cls.load_config(config.renderer.checkpoint_path, **load_kwargs)
			scheduler = scheduler_cls.from_config(scheduler_config)
		else:
			scheduler = scheduler_cls()
	except Exception as error:
		import warnings
		warnings.warn(
			f"Failed to load Wan scheduler config from {config.renderer.checkpoint_path!r}: {error}. "
			"Falling back to the diffusers default scheduler.",
			RuntimeWarning,
		)
		scheduler = scheduler_cls()

	if config.renderer.train_scheduler_steps is None:
		return scheduler

	train_scheduler_steps = int(config.renderer.train_scheduler_steps)
	if int(scheduler.config.num_train_timesteps) == train_scheduler_steps:
		return scheduler

	return scheduler_cls.from_config(scheduler.config, num_train_timesteps=train_scheduler_steps)


class DeepWorldQW(nn.Module):
	"""Top-level grounded video model combining a Qwen brain and Wan renderer.

	Args:
		config: Root configuration object describing all model and training settings.
	"""

	def __init__(self, config: DeepWorldQWConfig):
		super().__init__()
		self.config = config
		self.brain = DeepWorldQWBrain(
			config.brain,
			gradient_checkpointing=config.training.gradient_checkpointing,
		)
		self.renderer = DeepWorldQWRenderer(
			config.renderer,
			condition_dim=self.brain.hidden_size,
			gradient_checkpointing=config.training.gradient_checkpointing,
		)
		self.scheduler = build_wan_scheduler(config, FlowMatchEulerDiscreteScheduler)

	def _sample_condition_drop(self, device: torch.device) -> Tensor | bool:
		"""Sample the scalar classifier-free conditioning dropout flag."""

		dropout_prob = self.config.renderer.condition_dropout_prob
		if not self.training or dropout_prob <= 0.0:
			return False
		if dropout_prob >= 1.0:
			return True
		return torch.rand((), device=device) < dropout_prob

	def forward(
		self,
		batch: dict[str, Any],
		return_auxiliary: bool = False,
		generate_sample: bool = False,
		generator: torch.Generator | None = None,
	) -> dict[str, Tensor]:
		"""Run one training forward pass and return latent-space denoising loss.

		Args:
			batch: Collated training batch produced by the dataset pipeline.
			return_auxiliary: Whether to include non-loss tensors in the return payload.
			generate_samples: Whether to run the inference path instead of training loss computation.
			generator: Optional random generator used by the inference path.

		Returns:
			A dictionary containing the scalar loss, generated videos, or optional auxiliary tensors.
		"""

		if generate_sample:
			return {"video": self.generate(batch, generator=generator)}

		video = batch["video"].unsqueeze(0).to(device=next(self.parameters()).device, non_blocking=True)
		latents = self.renderer.encode_videos(video, sample_posterior=self.config.renderer.vae_sample_posterior)
		latent_patch_grid = self.renderer.latent_patch_grid(latents)

		brain_outputs = self.brain(batch, latent_patch_grid=latent_patch_grid)
		condition_hidden_states = brain_outputs["gen_hidden_states"]

		noise = torch.randn_like(latents)
		sigma = torch.rand((), device=latents.device, dtype=latents.dtype)
		sigma_view = sigma.view(1, 1, 1, 1, 1)
		noisy_latents = sigma_view * noise + (1.0 - sigma_view) * latents
		target = noise - latents

		model_output = self.renderer(
			hidden_states=noisy_latents,
			timestep=sigma * self.scheduler.config.num_train_timesteps,
			condition_hidden_states=condition_hidden_states,
			drop_condition=self._sample_condition_drop(latents.device),
			return_dict=True,
		)["sample"]

		loss = (model_output.float() - target.float()).pow(2).mean()

		if not return_auxiliary:
			return {"loss": loss}
		return {
			"loss": loss,
			"pred": model_output,
			"target": target,
			"latents": latents,
		}

	@torch.no_grad()
	def generate(
		self,
		batch: dict[str, Any],
		num_frames: int | None = None,
		height: int | None = None,
		width: int | None = None,
		num_inference_steps: int | None = None,
		generator: torch.Generator | None = None,
	) -> Tensor:
		"""Generate a video sample from prompts and reference images.

		Args:
			batch: Collated batch containing prompts and reference images.
			num_frames: Optional output frame count. If omitted, taken from config.
			height: Optional output height. If omitted, taken from config.
			width: Optional output width. If omitted, taken from config.
			num_inference_steps: Optional number of scheduler steps.
			generator: Optional torch random generator.

		Returns:
			A decoded video tensor generated from random Wan latents and Qwen conditioning.
		"""

		device = next(self.parameters()).device
		num_frames = num_frames or self.config.dataset.video_num_frames
		height = height or self.config.dataset.video_height
		width = width or self.config.dataset.video_width
		num_frames = (
			(num_frames - 1)
			// self.renderer.vae.config.scale_factor_temporal
			* self.renderer.vae.config.scale_factor_temporal
			+ 1
		)

		latent_frames = (num_frames - 1) // self.renderer.vae.config.scale_factor_temporal + 1
		latents = torch.randn(
			self.renderer.transformer.config.in_channels,
			latent_frames,
			height // self.renderer.vae.config.scale_factor_spatial,
			width // self.renderer.vae.config.scale_factor_spatial,
			device=device,
			dtype=self.renderer.transformer.patch_embedding.weight.dtype,
			generator=generator,
		).unsqueeze(0)

		latent_patch_grid = self.renderer.latent_patch_grid(latents)
		brain_outputs = self.brain(batch, latent_patch_grid=latent_patch_grid)
		self.scheduler.set_timesteps(num_inference_steps or self.config.renderer.inference_steps, device=device)

		for timestep in self.scheduler.timesteps:
			model_output = self.renderer(
				hidden_states=latents,
				timestep=timestep,
				condition_hidden_states=brain_outputs["gen_hidden_states"],
				return_dict=True,
			)["sample"]
			latents = self.scheduler.step(model_output, timestep, latents, return_dict=False)[0]

		return self.renderer.decode_latents(latents)[0]
