import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.config import DeepWorldHYConfig, DeepWorldHYModelConfig
from src.modules import inject_lora_layers
from src.utils import resolve_torch_dtype

from hyvideo.hyvideo.commons.parallel_states import get_parallel_state
from hyvideo.hyvideo.models.autoencoders import hunyuanvideo_15_vae
from hyvideo.hyvideo.models.transformers.hunyuanvideo_1_5_transformer import (
	HunyuanVideo_1_5_DiffusionTransformer,
)
from hyvideo.hyvideo.models.transformers.modules.attention import parallel_attention
from hyvideo.hyvideo.models.transformers.modules.modulate_layers import apply_gate, modulate
from hyvideo.hyvideo.models.transformers.modules.posemb_layers import apply_rotary_emb, get_1d_rotary_pos_embed
from hyvideo.hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hyvideo.hyvideo.utils.communications import all_gather
from vggt.vggt.models.vggt import VGGT


def _reset_module_parameters(module: nn.Module) -> None:
	"""Best-effort parameter reset for freshly initialized copied modules."""

	for child in module.modules():
		reset_parameters = getattr(child, "reset_parameters", None)
		if callable(reset_parameters):
			reset_parameters()


STREAM_MODULE_SUFFIXES = [
	"mod",
	"norm1",
	"attn_q",
	"attn_k",
	"attn_v",
	"attn_q_norm",
	"attn_k_norm",
	"attn_proj",
	"norm2",
	"mlp",
]


def _build_stream_modules(
	base_block: nn.Module,
	prefix: str,
	clone_modules: bool,
	init_mode: str = "copy_image",
) -> nn.ModuleDict:
	"""Build a stream module table from a Hunyuan double-stream block.

	Args:
		base_block: Pretrained Hunyuan double-stream block.
		prefix: Source stream prefix, either `img` or `txt`.
		clone_modules: Whether modules are deep-copied for a new trainable stream.
		init_mode: `fresh` resets cloned parameters after copying.

	Returns:
		A module dictionary keyed by role, independent of the source prefix.
	"""

	modules = nn.ModuleDict({
		suffix: deepcopy(getattr(base_block, f"{prefix}_{suffix}")) if clone_modules else getattr(base_block, f"{prefix}_{suffix}")
		for suffix in STREAM_MODULE_SUFFIXES
	})
	if init_mode == "fresh":
		_reset_module_parameters(modules)
	if clone_modules:
		modules.requires_grad_(True)
	return modules


class MMTripleStreamBlock(nn.Module):
	"""Hunyuan double-stream block extended with a trainable geometry stream.

	The pretrained image and text branches are reused by reference. The geometry
	branch owns its own modulation, attention, and MLP modules. Joint attention
	is still computed over all image, geometry, and text tokens by concatenating
	image and geometry queries during the attention call and splitting the result
	afterward.
	"""

	def __init__(self, base_block: nn.Module, geo_stream_init: str):
		"""Wrap one pretrained Hunyuan double-stream block."""

		super().__init__()
		self.heads_num = base_block.heads_num
		self.attn_mode = base_block.attn_mode
		self.hybrid_seq_parallel_attn = None

		self.img = _build_stream_modules(base_block, "img", clone_modules=False)
		self.txt = _build_stream_modules(base_block, "txt", clone_modules=False)
		self.geo = _build_stream_modules(base_block, "img", clone_modules=True, init_mode=geo_stream_init)

	def _project_stream_attention(
		self,
		hidden_states: Tensor,
		modules: dict[str, nn.Module],
		vec: Tensor,
		freqs_cis: tuple[Tensor, Tensor] | None,
	) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
		"""Apply one stream's attention AdaLN, QKV projection, and RoPE."""

		mod1_shift, mod1_scale, mod1_gate, mod2_shift, mod2_scale, mod2_gate = modules["mod"](vec).chunk(6, dim=-1)
		modulated = modulate(modules["norm1"](hidden_states), shift=mod1_shift, scale=mod1_scale)

		query = modules["attn_q"](modulated)
		key = modules["attn_k"](modulated)
		value = modules["attn_v"](modulated)
		query = rearrange(query, "B L (H D) -> B L H D", H=self.heads_num)
		key = rearrange(key, "B L (H D) -> B L H D", H=self.heads_num)
		value = rearrange(value, "B L (H D) -> B L H D", H=self.heads_num)
		query = modules["attn_q_norm"](query).to(value)
		key = modules["attn_k_norm"](key).to(value)

		if freqs_cis is not None and hidden_states.size(1) > 0:
			query, key = apply_rotary_emb(query, key, freqs_cis, head_first=False)

		return query, key, value, mod1_gate, mod2_shift, mod2_scale, mod2_gate

	def _apply_stream_mlp(
		self,
		hidden_states: Tensor,
		modules: dict[str, nn.Module],
		mod2_shift: Tensor,
		mod2_scale: Tensor,
		mod2_gate: Tensor,
	) -> Tensor:
		"""Apply the official post-attention AdaLN MLP update for one stream."""

		mlp_input = modulate(modules["norm2"](hidden_states), shift=mod2_shift, scale=mod2_scale)
		return hidden_states + apply_gate(modules["mlp"](mlp_input), gate=mod2_gate)

	def forward(
		self,
		img: Tensor,
		geo: Tensor,
		txt: Tensor,
		vec: Tensor,
		img_freqs: tuple[Tensor, Tensor] | None,
		geo_freqs: tuple[Tensor, Tensor] | None,
		txt_freqs: tuple[Tensor, Tensor] | None,
		text_mask: Tensor | None = None,
		attn_param: dict[str, Any] | None = None,
		is_flash: bool = False,
		block_idx: int | None = None,
	) -> tuple[Tensor, Tensor, Tensor]:
		"""Run one three-stream MMDiT block."""

		img_q, img_k, img_v, img_gate, img_mlp_shift, img_mlp_scale, img_mlp_gate = self._project_stream_attention(
			img, self.img, vec, img_freqs,
		)
		geo_q, geo_k, geo_v, geo_gate, geo_mlp_shift, geo_mlp_scale, geo_mlp_gate = self._project_stream_attention(
			geo, self.geo, vec, geo_freqs,
		)
		txt_q, txt_k, txt_v, txt_gate, txt_mlp_shift, txt_mlp_scale, txt_mlp_gate = self._project_stream_attention(
			txt, self.txt, vec, txt_freqs,
		)

		prefix_q = torch.cat([img_q, geo_q], dim=1)
		prefix_k = torch.cat([img_k, geo_k], dim=1)
		prefix_v = torch.cat([img_v, geo_v], dim=1)
		
		attn_mode = "flash" if is_flash else self.attn_mode
		attn = parallel_attention(
			(prefix_q, txt_q),
			(prefix_k, txt_k),
			(prefix_v, txt_v),
			img_q_len=prefix_q.shape[1],
			img_kv_len=prefix_k.shape[1],
			text_mask=text_mask,
			attn_mode=attn_mode,
			attn_param=attn_param,
			block_idx=block_idx,
		)

		img_attn, geo_attn, txt_attn = attn.split([img_q.shape[1], geo_q.shape[1], txt_q.shape[1]], dim=1)
		img = img + apply_gate(self.img["attn_proj"](img_attn), gate=img_gate)
		geo = geo + apply_gate(self.geo["attn_proj"](geo_attn), gate=geo_gate)
		txt = txt + apply_gate(self.txt["attn_proj"](txt_attn), gate=txt_gate)

		img = self._apply_stream_mlp(img, self.img, img_mlp_shift, img_mlp_scale, img_mlp_gate)
		geo = self._apply_stream_mlp(geo, self.geo, geo_mlp_shift, geo_mlp_scale, geo_mlp_gate)
		txt = self._apply_stream_mlp(txt, self.txt, txt_mlp_shift, txt_mlp_scale, txt_mlp_gate)
		return img, geo, txt


class MMSingleStreamBlock(nn.Module):
	"""Hunyuan single-stream block with MRoPE applied to every modality span."""

	def __init__(self, base_block: nn.Module):
		"""Wrap one pretrained Hunyuan single-stream block."""

		super().__init__()
		self.attn_mode = base_block.attn_mode
		self.hidden_size = base_block.hidden_size
		self.heads_num = base_block.heads_num
		self.mlp_hidden_dim = base_block.mlp_hidden_dim
		self.scale = base_block.scale
		self.linear1_q = base_block.linear1_q
		self.linear1_k = base_block.linear1_k
		self.linear1_v = base_block.linear1_v
		self.linear1_mlp = base_block.linear1_mlp
		self.linear2 = base_block.linear2
		self.mlp_act = base_block.mlp_act
		self.q_norm = base_block.q_norm
		self.k_norm = base_block.k_norm
		self.pre_norm = base_block.pre_norm
		self.modulation = base_block.modulation
		self.hybrid_seq_parallel_attn = None

	def _apply_span_rope(
		self,
		query: Tensor,
		key: Tensor,
		start: int,
		length: int,
		freqs_cis: tuple[Tensor, Tensor] | None,
	) -> tuple[Tensor, Tensor]:
		"""Apply RoPE to one contiguous span inside a joined sequence."""

		if length == 0 or freqs_cis is None:
			return query[:, start:(start + length)], key[:, start:(start + length)]
		span_q = query[:, start:(start + length)]
		span_k = key[:, start:(start + length)]
		return apply_rotary_emb(span_q, span_k, freqs_cis, head_first=False)

	def forward(
		self,
		x: Tensor,
		vec: Tensor,
		img_len: int,
		geo_len: int,
		txt_len: int,
		img_freqs: tuple[Tensor, Tensor] | None,
		geo_freqs: tuple[Tensor, Tensor] | None,
		txt_freqs: tuple[Tensor, Tensor] | None,
		text_mask: Tensor | None = None,
		attn_param: dict[str, Any] | None = None,
		is_flash: bool = False,
	) -> Tensor:
		"""Run one single-stream block over image, geometry, and text tokens."""

		mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
		x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)

		query = self.linear1_q(x_mod)
		key = self.linear1_k(x_mod)
		value = self.linear1_v(x_mod)
		query = rearrange(query, "B L (H D) -> B L H D", H=self.heads_num)
		key = rearrange(key, "B L (H D) -> B L H D", H=self.heads_num)
		value = rearrange(value, "B L (H D) -> B L H D", H=self.heads_num)
		mlp = self.linear1_mlp(x_mod)

		query = self.q_norm(query).to(value)
		key = self.k_norm(key).to(value)

		img_q, img_k = self._apply_span_rope(query, key, 0, img_len, img_freqs)
		geo_q, geo_k = self._apply_span_rope(query, key, img_len, geo_len, geo_freqs)
		txt_q, txt_k = self._apply_span_rope(query, key, img_len + geo_len, txt_len, txt_freqs)

		img_v = value[:, :img_len]
		geo_v = value[:, img_len:(img_len + geo_len)]
		txt_v = value[:, (img_len + geo_len):(img_len + geo_len + txt_len)]
		prefix_q = torch.cat([img_q, geo_q], dim=1)
		prefix_k = torch.cat([img_k, geo_k], dim=1)
		prefix_v = torch.cat([img_v, geo_v], dim=1)

		attn_mode = "flash" if is_flash else self.attn_mode
		attn = parallel_attention(
			(prefix_q, txt_q),
			(prefix_k, txt_k),
			(prefix_v, txt_v),
			img_q_len=prefix_q.shape[1],
			img_kv_len=prefix_k.shape[1],
			text_mask=text_mask,
			attn_mode=attn_mode,
			attn_param=attn_param,
		)
		output = self.linear2(attn, self.mlp_act(mlp))
		return x + apply_gate(output, gate=mod_gate)


class DeepWorldHY(nn.Module):
	"""VGGT-conditioned three-stream world model built on HunyuanVideo-1.5."""

	def __init__(self, config: DeepWorldHYConfig):
		"""Load frozen encoders and install the trainable geometry stream."""

		super().__init__()
		self.config = config
		self.hy_config = config.model

		transformer_dtype = resolve_torch_dtype(self.hy_config.transformer_dtype)
		transformer_path = Path(self.hy_config.checkpoint_path) / "transformer" / self.hy_config.transformer_version
		transformer_kwargs: dict[str, Any] = {
			"pretrained_model_name_or_path": str(transformer_path),
			"low_cpu_mem_usage": True,
			"local_files_only": True,
		}
		if transformer_dtype is not None:
			transformer_kwargs["torch_dtype"] = transformer_dtype
		self.transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(**transformer_kwargs)
		self.transformer.set_attn_mode(self.hy_config.attention_mode)
		self.transformer.requires_grad_(False)

		self._install_three_stream_blocks(self.hy_config)
		self._inject_lora_adapters(self.hy_config)

		vae_dtype = resolve_torch_dtype(self.hy_config.vae_dtype)
		vae_kwargs: dict[str, Any] = {}
		if vae_dtype is not None:
			vae_kwargs["torch_dtype"] = vae_dtype
		self.vae = hunyuanvideo_15_vae.AutoencoderKLConv3D.from_pretrained(
			str(Path(self.hy_config.checkpoint_path) / "vae"),
			**vae_kwargs,
		)
		self.vae.requires_grad_(False)

		self.txt_encoder, _ = HunyuanVideo_1_5_Pipeline._load_text_encoders(
			self.hy_config.checkpoint_path,
			device=None,
		)
		self.txt_encoder.requires_grad_(False)
		self.vis_encoder = HunyuanVideo_1_5_Pipeline._load_vision_encoder(
			self.hy_config.checkpoint_path,
			device=None,
		)
		self.vis_encoder.requires_grad_(False)

		self.geo_encoder = VGGT.from_pretrained(
			self.hy_config.vggt_checkpoint_path,
			local_files_only=True,
			enable_camera=False,
			enable_point=False,
			enable_depth=False,
			enable_track=False,
		)
		vggt_dtype = resolve_torch_dtype(self.hy_config.vggt_dtype)
		if vggt_dtype is not None:
			self.geo_encoder.to(dtype=vggt_dtype)
		self.geo_encoder.requires_grad_(False)

		trainable_dtype = next(self.transformer.parameters()).dtype
		self.geo_patch_size = self.geo_encoder.aggregator.patch_size
		geo_hidden_size = self.geo_encoder.aggregator.frame_blocks[0].norm1.weight.size(0) * 2
		self.geo_in = nn.Sequential(
			nn.LayerNorm(geo_hidden_size, dtype=trainable_dtype),
			nn.Linear(geo_hidden_size, self.transformer.hidden_size, dtype=trainable_dtype),
			nn.GELU(),
			nn.Linear(self.transformer.hidden_size, self.transformer.hidden_size, dtype=trainable_dtype),
			nn.LayerNorm(self.transformer.hidden_size, dtype=trainable_dtype),
		)

		vis_in = getattr(self.transformer, "vision_in", None)
		if vis_in is None:
			vision_hidden_size = int(self.vis_encoder.model.config.hidden_size)
			self.vis_in = nn.Sequential(
				nn.LayerNorm(vision_hidden_size, dtype=trainable_dtype),
				nn.Linear(vision_hidden_size, self.transformer.hidden_size, dtype=trainable_dtype),
				nn.GELU(),
				nn.Linear(self.transformer.hidden_size, self.transformer.hidden_size, dtype=trainable_dtype),
				nn.LayerNorm(self.transformer.hidden_size, dtype=trainable_dtype),
			)
		else:
			self.vis_in = vis_in

		self.modality_embeddings = nn.ParameterDict({
			"video": nn.Parameter(torch.empty(1, 1, self.transformer.hidden_size, dtype=trainable_dtype)),
			"vae": nn.Parameter(torch.empty(1, 1, self.transformer.hidden_size, dtype=trainable_dtype)),
			"geo": nn.Parameter(torch.empty(1, 1, self.transformer.hidden_size, dtype=trainable_dtype)),
			"vis": nn.Parameter(torch.empty(1, 1, self.transformer.hidden_size, dtype=trainable_dtype)),
			"txt": nn.Parameter(torch.empty(1, 1, self.transformer.hidden_size, dtype=trainable_dtype)),
		})
		for embedding in self.modality_embeddings.values():
			nn.init.zeros_(embedding)
		self._set_frozen_modules_eval()

	def _set_frozen_modules_eval(self) -> None:
		"""Keep frozen encoders in inference mode even during training."""

		self.vae.eval()
		self.txt_encoder.eval()
		self.vis_encoder.eval()
		self.geo_encoder.eval()

	def train(self, mode: bool = True):
		"""Set train mode while preserving eval mode for frozen encoders."""

		super().train(mode)
		self._set_frozen_modules_eval()
		return self

	def _install_three_stream_blocks(self, hy_config: DeepWorldHYModelConfig) -> None:
		"""Replace Hunyuan blocks with MRoPE-aware wrapper blocks."""

		self.transformer.double_blocks = nn.ModuleList([
			MMTripleStreamBlock(block, geo_stream_init=hy_config.geo_stream_init)
			for block in self.transformer.double_blocks
		])
		self.transformer.single_blocks = nn.ModuleList([
			MMSingleStreamBlock(block)
			for block in self.transformer.single_blocks
		])

	def _inject_lora_adapters(self, hy_config: DeepWorldHYModelConfig) -> None:
		"""Install LoRA on frozen pretrained streams without wrapping the geo stream."""

		target_names = set(hy_config.lora_target_modules)

		def stream_targets(prefix: str) -> set[str]:
			targets: set[str] = set()
			for role in ("attn_q", "attn_k", "attn_v", "attn_proj"):
				if role in target_names or f"{prefix}_{role}" in target_names:
					targets.add(role)
			return targets

		for block in self.transformer.double_blocks:
			inject_lora_layers(
				block.img,
				target_names=stream_targets("img"),
				rank=hy_config.lora_rank,
				alpha=hy_config.lora_alpha,
				dropout=hy_config.lora_dropout,
			)
			inject_lora_layers(
				block.txt,
				target_names=stream_targets("txt"),
				rank=hy_config.lora_rank,
				alpha=hy_config.lora_alpha,
				dropout=hy_config.lora_dropout,
			)

		for block in self.transformer.single_blocks:
			inject_lora_layers(
				block,
				target_names=target_names,
				rank=hy_config.lora_rank,
				alpha=hy_config.lora_alpha,
				dropout=hy_config.lora_dropout,
			)

	def _split_for_sequence_parallel(
		self,
		tokens: Tensor,
		freqs_cis: tuple[Tensor, Tensor] | None,
		stream_name: str,
	) -> tuple[Tensor, tuple[Tensor, Tensor] | None, int, int]:
		"""Shard one prefix stream along sequence length for Hunyuan sequence parallelism."""

		parallel_dims = get_parallel_state()
		if not getattr(parallel_dims, "sp_enabled", False):
			return tokens, freqs_cis, 0, tokens.size(1)

		sp_size = parallel_dims.sp
		sp_rank = parallel_dims.sp_rank
		token_count = tokens.size(1)
		if token_count == 0:
			return tokens, freqs_cis, 0, 0
		if token_count % sp_size != 0:
			min_supported_tokens = (token_count // sp_size + 1) * (sp_size - 1)
			if token_count <= min_supported_tokens:
				raise ValueError(
					f"`{stream_name}` has {token_count} tokens, which is too short for "
					f"sequence_parallel_size={sp_size}. Increase token count or reduce SP size."
				)

		token_chunks = torch.chunk(tokens, sp_size, dim=1)
		if len(token_chunks) != sp_size:
			raise ValueError(
				f"`{stream_name}` produced {len(token_chunks)} sequence chunks for "
				f"sequence_parallel_size={sp_size}; reduce SP size or increase token count."
			)
		local_start = sum(chunk.size(1) for chunk in token_chunks[:sp_rank])
		local_end = local_start + token_chunks[sp_rank].size(1)

		if freqs_cis is None:
			return token_chunks[sp_rank], None, local_start, local_end
		cos_chunks = torch.chunk(freqs_cis[0], sp_size, dim=0)
		sin_chunks = torch.chunk(freqs_cis[1], sp_size, dim=0)
		if len(cos_chunks) != sp_size or len(sin_chunks) != sp_size:
			raise ValueError(
				f"`{stream_name}` rotary positions cannot be split into "
				f"{sp_size} sequence-parallel chunks."
			)
		return token_chunks[sp_rank], (cos_chunks[sp_rank], sin_chunks[sp_rank]), local_start, local_end

	@property
	def vae_spatial_compression_ratio(self) -> int:
		"""Return the Hunyuan VAE spatial compression ratio."""

		return int(getattr(self.vae.config, "ffactor_spatial", 16))

	@property
	def vae_temporal_compression_ratio(self) -> int:
		"""Return the Hunyuan VAE temporal compression ratio."""

		return int(getattr(self.vae.config, "ffactor_temporal", 4))

	def _scale_vae_latents(self, latents: Tensor) -> Tensor:
		"""Apply Hunyuan VAE scaling and optional shift."""

		scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))
		shift_factor = getattr(self.vae.config, "shift_factor", None)
		if shift_factor:
			return (latents - shift_factor) * scaling_factor
		return latents * scaling_factor

	def _unscale_vae_latents(self, latents: Tensor) -> Tensor:
		"""Invert Hunyuan VAE scaling and optional shift."""

		scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))
		latents = latents / scaling_factor
		shift_factor = getattr(self.vae.config, "shift_factor", None)
		if shift_factor:
			latents = latents + shift_factor
		return latents

	def encode_vae(self, pixel_values: Tensor, sample_posterior: bool = False) -> Tensor:
		"""Encode videos or single-frame images into scaled Hunyuan latent space."""

		vae_param = next(self.vae.parameters())
		pixel_values = pixel_values.to(device=vae_param.device, dtype=vae_param.dtype, non_blocking=True)
		if pixel_values.dim() == 4:
			pixel_values = pixel_values.unsqueeze(2)
		with torch.no_grad(), self.vae.memory_efficient_context():
			posterior = self.vae.encode(pixel_values).latent_dist
			latents = posterior.sample() if sample_posterior else posterior.mode()
		return self._scale_vae_latents(latents).to(next(self.transformer.parameters()).dtype)

	def decode_latents(self, latents: Tensor) -> Tensor:
		"""Decode scaled Hunyuan latents into video pixel space."""

		vae_param = next(self.vae.parameters())
		latents = self._unscale_vae_latents(latents).to(device=vae_param.device, dtype=vae_param.dtype)
		with torch.no_grad(), self.vae.memory_efficient_context():
			return self.vae.decode(latents, return_dict=False)[0]

	def _video_condition_channels(self, latents: Tensor) -> Tensor:
		"""Build empty i2v condition channels for target video tokens."""

		mask_channel = torch.zeros(
			latents.size(0), 1, *latents.shape[-3:],
			device=latents.device,
			dtype=latents.dtype,
		)
		ref_latents = torch.zeros_like(latents)
		return torch.cat([latents, ref_latents, mask_channel], dim=1)

	def _reference_condition_channels(self, ref_latents: Tensor) -> Tensor:
		"""Build i2v-style condition channels for reference image latents."""

		mask_channel = torch.ones(
			ref_latents.size(0), 1, *ref_latents.shape[-3:],
			device=ref_latents.device,
			dtype=ref_latents.dtype,
		)
		latents = torch.zeros_like(ref_latents)
		return torch.cat([latents, ref_latents, mask_channel], dim=1)

	def _latent_patch_grid(self, latents: Tensor) -> tuple[int, int, int]:
		"""Return the Hunyuan patch grid for a latent tensor."""

		patch_t, patch_h, patch_w = self.transformer.patch_size
		latent_t, latent_h, latent_w = latents.shape[-3:]
		if latent_t % patch_t != 0 or latent_h % patch_h != 0 or latent_w % patch_w != 0:
			raise ValueError(
				"Hunyuan latent shape must be divisible by transformer patch size; "
				f"latent={(latent_t, latent_h, latent_w)}, patch={self.transformer.patch_size}."
			)
		return latent_t // patch_t, latent_h // patch_h, latent_w // patch_w

	def _make_grid_positions(self, grid: tuple[int, int, int], start_index: int, device: torch.device) -> tuple[Tensor, int]:
		"""Build `(t, h, w)` MRoPE positions for one grid-shaped span."""

		grid_t, grid_h, grid_w = grid
		t_index = torch.arange(grid_t, device=device).view(-1, 1, 1).expand(-1, grid_h, grid_w).flatten()
		h_index = torch.arange(grid_h, device=device).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
		w_index = torch.arange(grid_w, device=device).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
		positions = torch.stack([t_index, h_index, w_index], dim=1) + start_index
		next_index = int(positions.max().item()) + 1 if positions.numel() > 0 else start_index
		return positions, next_index

	def _make_text_positions(self, length: int, start_index: int, device: torch.device) -> tuple[Tensor, int]:
		"""Build diagonal MRoPE positions for text-style token spans."""

		positions = torch.arange(start_index, start_index + length, device=device, dtype=torch.long)
		return positions.unsqueeze(1).expand(-1, 3), start_index + length

	def _build_mrope_freqs(self, positions: Tensor) -> tuple[Tensor, Tensor] | None:
		"""Convert explicit three-axis positions into Hunyuan RoPE cos/sin tensors."""

		if positions.size(0) == 0:
			return None
		cos_parts: list[Tensor] = []
		sin_parts: list[Tensor] = []
		for axis, dim in enumerate(self.transformer.rope_dim_list):
			cos, sin = get_1d_rotary_pos_embed(
				dim,
				positions[:, axis].float().cpu(),
				theta=self.transformer.rope_theta,
				use_real=True,
			)
			cos_parts.append(cos.to(device=positions.device))
			sin_parts.append(sin.to(device=positions.device))
		return torch.cat(cos_parts, dim=1), torch.cat(sin_parts, dim=1)

	def _encode_text(self, prompt: str, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
		"""Encode prompt text with Hunyuan's frozen Qwen/LLM text encoder."""

		with torch.no_grad():
			text_inputs = self.txt_encoder.text2tokens(
				[prompt],
				data_type="video",
				max_length=self.config.dataset.max_text_length,
			)
			text_outputs = self.txt_encoder.encode(text_inputs, data_type="video", device=device)
		return text_outputs.hidden_state.to(device=device, dtype=dtype), text_outputs.attention_mask.to(device)

	def _encode_vis(self, vis_ref_images: Tensor) -> tuple[Tensor, int]:
		"""Encode reference images with frozen SigLIP and project them to Hunyuan hidden size."""

		if not self.hy_config.use_vis_tokens:
			empty = torch.empty(1, 0, self.transformer.hidden_size, device=vis_ref_images.device)
			return empty, 0

		if vis_ref_images.numel() == 0:
			empty = torch.empty(1, 0, self.transformer.hidden_size, device=vis_ref_images.device)
			return empty, 0
		images_np = (vis_ref_images.detach().float().cpu().permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype("uint8")
		with torch.no_grad():
			vision_states = self.vis_encoder.encode_images(images_np).last_hidden_state
		projector_dtype = next(self.vis_in.parameters()).dtype
		vis_tokens = self.vis_in(vision_states.to(device=vis_ref_images.device, dtype=projector_dtype))
		vis_tokens = vis_tokens.view(1, -1, vis_tokens.size(-1))
		return vis_tokens, int(vision_states.size(1))

	def _encode_geo(self, geo_ref_images: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
		"""Encode reference images with frozen VGGT and project geometry tokens."""

		if not self.hy_config.use_geo_tokens:
			empty = torch.empty(1, 0, self.transformer.hidden_size, device=geo_ref_images.device)
			return empty, (1, 0, 0)

		device = geo_ref_images.device
		vggt_dtype = next(self.geo_encoder.aggregator.parameters()).dtype
		with torch.no_grad():
			aggregated_tokens, patch_start_idx = self.geo_encoder.aggregator(geo_ref_images.unsqueeze(0).to(dtype=vggt_dtype))
			geo_tokens = aggregated_tokens[-1][0, :, patch_start_idx:, :]
		projector_dtype = next(self.geo_in.parameters()).dtype
		geo_tokens = self.geo_in(geo_tokens.to(dtype=projector_dtype))
		valid_tokens = geo_tokens.view(1, -1, geo_tokens.size(-1))

		grid_h = geo_ref_images.size(-2) // self.geo_patch_size
		grid_w = geo_ref_images.size(-1) // self.geo_patch_size
		return valid_tokens.to(device=device), (1, grid_h, grid_w)

	def _encode_vae_refs(self, vae_ref_images: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
		"""Encode reference images through the frozen VAE and Hunyuan image patch embedder."""

		if not self.hy_config.use_vae_tokens:
			empty = torch.empty(1, 0, self.transformer.hidden_size, device=vae_ref_images.device)
			return empty, (1, 0, 0)

		if vae_ref_images.numel() == 0:
			empty = torch.empty(1, 0, self.transformer.hidden_size, device=vae_ref_images.device)
			return empty, (1, 0, 0)

		ref_latents = self.encode_vae(vae_ref_images, sample_posterior=False)
		ref_input = self._reference_condition_channels(ref_latents)
		ref_tokens = self.transformer.img_in(ref_input.to(dtype=next(self.transformer.parameters()).dtype))
		ref_tokens = ref_tokens.view(1, -1, ref_tokens.size(-1))
		return ref_tokens, self._latent_patch_grid(ref_latents)

	def _build_positions(
		self,
		video_grid: tuple[int, int, int],
		vae_grid: tuple[int, int, int],
		geo_grid: tuple[int, int, int],
		vis_tokens_per_ref: int,
		text_length: int,
		reference_count: int,
		device: torch.device,
	) -> tuple[tuple[Tensor, Tensor] | None, tuple[Tensor, Tensor] | None, tuple[Tensor, Tensor] | None]:
		"""Build MRoPE frequencies in video, VAE, geometry, visual, text order."""

		cursor = 0
		img_positions: list[Tensor] = []
		geo_positions: list[Tensor] = []
		txt_positions: list[Tensor] = []

		positions, cursor = self._make_grid_positions(video_grid, cursor, device)
		img_positions.append(positions)
		for _ in range(reference_count if self.hy_config.use_vae_tokens else 0):
			positions, cursor = self._make_grid_positions(vae_grid, cursor, device)
			img_positions.append(positions)
		for _ in range(reference_count if self.hy_config.use_geo_tokens else 0):
			positions, cursor = self._make_grid_positions(geo_grid, cursor, device)
			geo_positions.append(positions)
		for _ in range(reference_count if self.hy_config.use_vis_tokens else 0):
			positions, cursor = self._make_text_positions(vis_tokens_per_ref, cursor, device)
			txt_positions.append(positions)
		if self.hy_config.use_txt_tokens:
			positions, cursor = self._make_text_positions(text_length, cursor, device)
			txt_positions.append(positions)

		img_freqs = self._build_mrope_freqs(torch.cat(img_positions, dim=0))
		geo_freqs = self._build_mrope_freqs(torch.cat(geo_positions, dim=0)) if geo_positions else None
		txt_freqs = self._build_mrope_freqs(torch.cat(txt_positions, dim=0)) if txt_positions else None
		return img_freqs, geo_freqs, txt_freqs

	def _apply_condition_dropout(self, *spans: Tensor) -> tuple[Tensor, ...]:
		"""Zero non-video conditioning spans for classifier-free-style training."""

		dropout_prob = self.hy_config.condition_dropout_prob
		if not self.training or dropout_prob <= 0.0:
			return spans
		keep_condition = torch.rand((), device=spans[0].device) >= dropout_prob
		if keep_condition:
			return spans
		return tuple(torch.zeros_like(span) for span in spans)

	def _time_vector(self, timesteps: Tensor, dtype: torch.dtype) -> Tensor:
		"""Build the Hunyuan time and optional embedded-guidance vector."""

		vec = self.transformer.time_in(timesteps).to(dtype=dtype)
		if self.transformer.guidance_embed:
			guidance = torch.full(
				(timesteps.size(0),),
				float(self.hy_config.guidance),
				device=timesteps.device,
				dtype=timesteps.dtype,
			)
			vec = vec + self.transformer.guidance_in(guidance).to(dtype=dtype)
		return vec

	def predict_velocity(
		self,
		latents_noised: Tensor,
		timesteps: Tensor,
		batch: dict[str, Any],
	) -> Tensor:
		"""Run the Hunyuan three-stream transformer and predict flow velocity."""

		device = latents_noised.device
		dtype = next(self.transformer.parameters()).dtype
		video_input = self._video_condition_channels(latents_noised)
		video_grid = self._latent_patch_grid(latents_noised)
		video_token_count = math.prod(video_grid)

		img = self.transformer.img_in(video_input.to(dtype=dtype))
		img = img + self.modality_embeddings["video"].to(device=device, dtype=img.dtype)

		vae_tokens, vae_grid = self._encode_vae_refs(
			batch["vae_ref_images"].to(device=device, non_blocking=True),
		)
		vae_tokens = vae_tokens + self.modality_embeddings["vae"].to(device=device, dtype=img.dtype)

		geo_tokens, geo_grid = self._encode_geo(
			batch["geo_ref_images"].to(device=device, non_blocking=True),
		)
		geo_tokens = geo_tokens + self.modality_embeddings["geo"].to(device=device, dtype=img.dtype)

		text_states, text_mask = self._encode_text(batch["prompt"], device=device, dtype=dtype)
		if not self.hy_config.use_txt_tokens:
			text_states = text_states[:, :0]
			text_mask = text_mask[:, :0]
		if text_states.size(1) > 0:
			if self.transformer.text_projection == "linear":
				text_tokens = self.transformer.txt_in(text_states)
			else:
				text_tokens = self.transformer.txt_in(
					text_states,
					timesteps,
					text_mask if self.transformer.use_attention_mask else None,
				)
		else:
			text_tokens = torch.empty(1, 0, self.transformer.hidden_size, device=device, dtype=dtype)
		txt_tokens = text_tokens + self.modality_embeddings["txt"].to(device=device, dtype=img.dtype)

		vis_tokens, vis_tokens_per_ref = self._encode_vis(
			batch["vis_ref_images"].to(device=device, non_blocking=True),
		)
		vis_tokens = vis_tokens + self.modality_embeddings["vis"].to(device=device, dtype=img.dtype)
		vis_mask = torch.ones(vis_tokens.size()[:2], device=device, dtype=text_mask.dtype)

		vae_tokens, geo_tokens, vis_tokens, txt_tokens = self._apply_condition_dropout(
			vae_tokens,
			geo_tokens,
			vis_tokens,
			txt_tokens,
		)

		img = torch.cat([img, vae_tokens.to(dtype=img.dtype)], dim=1)
		txt = torch.cat([vis_tokens.to(dtype=img.dtype), txt_tokens.to(dtype=img.dtype)], dim=1)
		txt_mask = torch.cat([vis_mask, text_mask], dim=1)
		geo = geo_tokens.to(dtype=img.dtype)

		reference_count = int(batch["vis_ref_images"].size(0))
		img_freqs, geo_freqs, txt_freqs = self._build_positions(
			video_grid=video_grid,
			vae_grid=vae_grid,
			geo_grid=geo_grid,
			vis_tokens_per_ref=vis_tokens_per_ref,
			text_length=int(txt_tokens.size(1)),
			reference_count=reference_count,
			device=device,
		)
		img, img_freqs, _, _ = self._split_for_sequence_parallel(img, img_freqs, "image")
		geo, geo_freqs, _, _ = self._split_for_sequence_parallel(geo, geo_freqs, "geometry")

		self.transformer.attn_param["thw"] = list(video_grid)
		vec = self._time_vector(timesteps, dtype=img.dtype)
		
		for index, block in enumerate(self.transformer.double_blocks):
			self.transformer.attn_param["layer-name"] = f"world_double_block_{index + 1}"
			img, geo, txt = block(
				img=img,
				geo=geo,
				txt=txt,
				vec=vec,
				img_freqs=img_freqs,
				geo_freqs=geo_freqs,
				txt_freqs=txt_freqs,
				text_mask=txt_mask,
				attn_param=self.transformer.attn_param,
				is_flash=False,
				block_idx=index,
			)

		img_len, geo_len, txt_len = img.size(1), geo.size(1), txt.size(1)
		x = torch.cat([img, geo, txt], dim=1)
		
		for index, block in enumerate(self.transformer.single_blocks):
			self.transformer.attn_param["layer-name"] = f"world_single_block_{index + 1}"
			x = block(
				x=x,
				vec=vec,
				img_len=img_len,
				geo_len=geo_len,
				txt_len=txt_len,
				img_freqs=img_freqs,
				geo_freqs=geo_freqs,
				txt_freqs=txt_freqs,
				text_mask=txt_mask,
				attn_param=self.transformer.attn_param,
				is_flash=False,
			)

		img_tokens = x[:, :img_len]
		img_tokens = self.transformer.final_layer(img_tokens, vec)
		parallel_dims = get_parallel_state()
		if getattr(parallel_dims, "sp_enabled", False):
			img_tokens = all_gather(img_tokens, dim=1, group=parallel_dims.sp_group)
		video_tokens = img_tokens[:, :video_token_count]
		return self.transformer.unpatchify(video_tokens, *video_grid)

	def _sample_timestep(self, device: torch.device) -> Tensor:
		"""Sample one shifted flow-matching timestep for the local sample."""

		t0 = 1e-5
		t1 = 1.0 - 1e-5
		if self.hy_config.snr_type == "uniform":
			t = torch.rand((), device=device) * (t1 - t0) + t0
		elif self.hy_config.snr_type == "lognorm":
			u = torch.normal(mean=0.0, std=1.0, size=(), device=device)
			t = torch.sigmoid(u) * (t1 - t0) + t0
		elif self.hy_config.snr_type == "mix":
			u = torch.normal(mean=0.0, std=1.0, size=(), device=device)
			t_lognorm = torch.sigmoid(u) * (t1 - t0) + t0
			t_uniform = torch.rand((), device=device) * (t1 - t0) + t0
			mask = (torch.rand((), device=device) > 0.3).float()
			t = mask * t_lognorm + (1.0 - mask) * t_uniform
		elif self.hy_config.snr_type == "mode":
			u = torch.rand((), device=device)
			mode_scale = 1.29
			t = 1.0 - u - mode_scale * (torch.cos(math.pi * u / 2.0).pow(2) - 1.0 + u)
			t = t * (t1 - t0) + t0
		else:
			raise ValueError(f"Unsupported SNR type: {self.hy_config.snr_type!r}.")

		timesteps = t * self.hy_config.num_train_timesteps
		shift = self.hy_config.train_timestep_shift
		if shift != 1.0:
			timesteps_normalized = timesteps / self.hy_config.num_train_timesteps
			timesteps = (
				shift * timesteps_normalized
				/ (1.0 + (shift - 1.0) * timesteps_normalized)
				* self.hy_config.num_train_timesteps
			)
		return timesteps.view(1)

	def _aligned_generation_shape(
		self,
		num_frames: int | None,
		height: int | None,
		width: int | None,
	) -> tuple[int, int, int, int, int, int]:
		"""Resolve a VAE- and transformer-compatible generation shape."""

		num_frames = num_frames or self.config.dataset.video_num_frames
		height = height or self.config.dataset.video_height
		width = width or self.config.dataset.video_width
		num_frames = (
			(num_frames - 1)
			// self.vae_temporal_compression_ratio
			* self.vae_temporal_compression_ratio
			+ 1
		)
		if height % self.vae_spatial_compression_ratio != 0 or width % self.vae_spatial_compression_ratio != 0:
			raise ValueError(
				"Generation height and width must be divisible by the VAE spatial compression ratio; "
				f"got height={height}, width={width}, ratio={self.vae_spatial_compression_ratio}."
			)

		latent_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
		latent_height = height // self.vae_spatial_compression_ratio
		latent_width = width // self.vae_spatial_compression_ratio
		patch_t, patch_h, patch_w = self.transformer.patch_size
		if latent_frames % patch_t != 0 or latent_height % patch_h != 0 or latent_width % patch_w != 0:
			raise ValueError(
				"Generation latent shape must be divisible by the transformer patch size; "
				f"latent={(latent_frames, latent_height, latent_width)}, patch={self.transformer.patch_size}."
			)
		return num_frames, height, width, latent_frames, latent_height, latent_width

	def _build_validation_scheduler(self) -> FlowMatchDiscreteScheduler:
		"""Create the Hunyuan flow-matching scheduler used for evaluation."""

		return FlowMatchDiscreteScheduler(
			num_train_timesteps=self.hy_config.num_train_timesteps,
			shift=self.hy_config.validation_timestep_shift,
			reverse=True,
			solver="euler",
		)

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
		"""Generate decoded videos from prompts and reference images."""

		device = next(self.transformer.parameters()).device
		dtype = next(self.transformer.parameters()).dtype
		_, _, _, latent_frames, latent_height, latent_width = self._aligned_generation_shape(
			num_frames=num_frames,
			height=height,
			width=width,
		)
		latents = torch.randn(
			self.transformer.in_channels,
			latent_frames,
			latent_height,
			latent_width,
			device=device,
			dtype=dtype,
			generator=generator,
		).unsqueeze(0)

		scheduler = self._build_validation_scheduler()
		scheduler.set_timesteps(
			num_inference_steps or self.hy_config.inference_steps,
			device=device,
			n_tokens=math.prod(self._latent_patch_grid(latents)),
		)
		for timestep in scheduler.timesteps:
			model_input = scheduler.scale_model_input(latents, timestep)
			model_output = self.predict_velocity(
				model_input,
				timestep.view(1),
				batch,
			)
			latents = scheduler.step(model_output, timestep, latents, return_dict=False)[0].to(dtype=dtype)

		return self.decode_latents(latents)[0]

	def forward(
		self,
		batch: dict[str, Any],
		return_auxiliary: bool = False,
		generate_sample: bool = False,
		generator: torch.Generator | None = None,
	) -> dict[str, Tensor]:
		"""Run one Hunyuan/VGGT world-model training or generation step."""

		if generate_sample:
			return {"video": self.generate(batch, generator=generator)}

		device = next(self.transformer.parameters()).device
		video = batch["video"].unsqueeze(0).to(device=device, non_blocking=True)
		latents = self.encode_vae(video, sample_posterior=False)

		noise = torch.randn_like(latents)
		timesteps = self._sample_timestep(device=latents.device)
		timestep_view = (timesteps / self.hy_config.num_train_timesteps).view(1, 1, 1, 1, 1)
		latents_noised = (1.0 - timestep_view) * latents + timestep_view * noise
		target = noise - latents

		pred = self.predict_velocity(latents_noised, timesteps, batch)
		loss = (pred.float() - target.float()).pow(2).mean()

		if not return_auxiliary:
			return {"loss": loss}
		return {
			"loss": loss,
			"pred": pred,
			"target": target,
			"latents": latents,
		}
