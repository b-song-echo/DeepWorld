from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
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
from hyvideo.hyvideo.pipelines.pipeline_utils import rescale_noise_cfg
from hyvideo.hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hyvideo.hyvideo.utils.communications import all_gather
from vggt.vggt.models.vggt import VGGT


@dataclass(frozen=True)
class Span:
	"""Contiguous sequence span with attached RoPE frequencies or gate modality."""

	start: int
	length: int
	attachment: Any

	def shifted(self, offset: int) -> Span:
		"""Return this span with its start index shifted by `offset`."""

		return Span(self.start + offset, self.length, self.attachment)


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


def _build_kv_gates(hy_config: DeepWorldHYModelConfig, dtype: torch.dtype) -> nn.ParameterDict:
	"""Create trainable scalar K/V gates enabled by the HY config."""

	gates = nn.ParameterDict()
	for modality in ("vae", "geo", "vis", "txt"):
		stream_enabled = modality == "txt" or getattr(hy_config, f"use_{modality}_tokens")
		initial_value = getattr(hy_config, f"{modality}_kv_gate")
		if stream_enabled and initial_value is not None:
			gates[modality] = nn.Parameter(torch.full((), initial_value, dtype=dtype))
	return gates


def _apply_qk_rope(query: Tensor, key: Tensor, rope_spans: list[Span] | None) -> tuple[Tensor, Tensor]:
	"""Apply RoPE to selected contiguous spans while leaving other tokens untouched."""

	if not rope_spans:
		return query, key

	query_parts: list[Tensor] = []
	key_parts: list[Tensor] = []
	cursor = 0
	for span in sorted(rope_spans, key=lambda item: item.start):
		freqs_cis = span.attachment
		end = span.start + span.length
		if cursor < span.start:
			query_parts.append(query[:, cursor:span.start])
			key_parts.append(key[:, cursor:span.start])
		span_q, span_k = apply_rotary_emb(
			query[:, span.start:end], key[:, span.start:end],
			freqs_cis, head_first=False,
		)
		query_parts.append(span_q)
		key_parts.append(span_k)
		cursor = end
	if cursor < query.size(1):
		query_parts.append(query[:, cursor:])
		key_parts.append(key[:, cursor:])
	query = torch.cat(query_parts, dim=1)
	key = torch.cat(key_parts, dim=1)
	return query, key


def _apply_kv_gate(
	key: Tensor,
	value: Tensor,
	gate_spans: list[Span] | None,
	kv_gates: nn.ParameterDict,
) -> tuple[Tensor, Tensor]:
	"""Scale selected K/V spans with learnable modality gates."""

	if not gate_spans or len(kv_gates) == 0:
		return key, value

	key = key.clone()
	value = value.clone()
	for span in gate_spans:
		modality = span.attachment
		if modality not in kv_gates:
			continue
		gate = kv_gates[modality].to(device=key.device, dtype=key.dtype)
		if span.length == 0:
			continue
		end = span.start + span.length
		key[:, span.start:end] = key[:, span.start:end] * gate
		value[:, span.start:end] = value[:, span.start:end] * gate
	return key, value


class MMTripleStreamBlock(nn.Module):
	"""Hunyuan double-stream block optionally extended with a geometry stream.

	The pretrained image and text branches are reused by reference. When geometry
	is enabled, the geometry branch owns trainable copied modules and joins image
	tokens as an image-like prefix during attention. When geometry is disabled,
	the block remains a two-stream MMDiT block with no geometry modules.
	"""

	def __init__(self, base_block: nn.Module, hy_config: DeepWorldHYModelConfig):
		"""Wrap one pretrained Hunyuan double-stream block."""

		super().__init__()
		self.heads_num = base_block.heads_num
		self.attn_mode = base_block.attn_mode
		self.hybrid_seq_parallel_attn = None
		self.use_geo_stream = hy_config.use_geo_tokens

		self.img = _build_stream_modules(base_block, "img", clone_modules=False)
		self.txt = _build_stream_modules(base_block, "txt", clone_modules=False)
		if self.use_geo_stream:
			self.geo = _build_stream_modules(
				base_block, "img", clone_modules=True,
				init_mode=hy_config.geo_stream_init
			)
		self.kv_gates = _build_kv_gates(
			hy_config, next(base_block.parameters()).dtype
		)

	def _project_stream_attention(
		self,
		hidden_states: Tensor,
		modules: dict[str, nn.Module],
		vec: Tensor,
		rope_spans: list[Span] | None,
		gate_spans: list[Span] | None,
	) -> dict[str, Tensor]:
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

		query, key = _apply_qk_rope(query, key, rope_spans)
		key, value = _apply_kv_gate(key, value, gate_spans, self.kv_gates)

		return {
			"query": query,
			"key": key,
			"value": value,
			"attn_gate": mod1_gate,
			"mlp_shift": mod2_shift,
			"mlp_scale": mod2_scale,
			"mlp_gate": mod2_gate,
		}

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
		img_rope_spans: list[Span] | None,
		geo_rope_spans: list[Span] | None,
		txt_rope_spans: list[Span] | None,
		img_gate_spans: list[Span] | None,
		geo_gate_spans: list[Span] | None,
		txt_gate_spans: list[Span] | None,
		text_mask: Tensor | None = None,
		attn_param: dict[str, Any] | None = None,
		is_flash: bool = False,
		block_idx: int | None = None,
	) -> tuple[Tensor, Tensor, Tensor]:
		"""Run one three-stream MMDiT block."""

		img_attn_in = self._project_stream_attention(
			img, self.img, vec, img_rope_spans, img_gate_spans,
		)
		txt_attn_in = self._project_stream_attention(
			txt, self.txt, vec, txt_rope_spans, txt_gate_spans,
		)

		if self.use_geo_stream:
			geo_attn_in = self._project_stream_attention(
				geo, self.geo, vec, geo_rope_spans, geo_gate_spans,
			)
			prefix_q = torch.cat([img_attn_in["query"], geo_attn_in["query"]], dim=1)
			prefix_k = torch.cat([img_attn_in["key"], geo_attn_in["key"]], dim=1)
			prefix_v = torch.cat([img_attn_in["value"], geo_attn_in["value"]], dim=1)
		else:
			prefix_q = img_attn_in["query"]
			prefix_k = img_attn_in["key"]
			prefix_v = img_attn_in["value"]
		
		attn_mode = "flash" if is_flash else self.attn_mode
		attn_out = parallel_attention(
			(prefix_q, txt_attn_in["query"]),
			(prefix_k, txt_attn_in["key"]),
			(prefix_v, txt_attn_in["value"]),
			img_q_len=prefix_q.shape[1],
			img_kv_len=prefix_k.shape[1],
			text_mask=text_mask,
			attn_mode=attn_mode,
			attn_param=attn_param,
			block_idx=block_idx,
		)

		img_len = img_attn_in["query"].shape[1]
		txt_len = txt_attn_in["query"].shape[1],
		if self.use_geo_stream:
			geo_len = geo_attn_in["query"].shape[1]
			img_attn, geo_attn, txt_attn = attn_out.split([img_len, geo_len, txt_len], dim=1)
		else:
			img_attn, txt_attn = attn_out.split([img_len, txt_len], dim=1)
		
		img = img + apply_gate(self.img["attn_proj"](img_attn), gate=img_attn_in["attn_gate"])
		txt = txt + apply_gate(self.txt["attn_proj"](txt_attn), gate=txt_attn_in["attn_gate"])
		if self.use_geo_stream:
			geo = geo + apply_gate(self.geo["attn_proj"](geo_attn), gate=geo_attn_in["attn_gate"])

		img = self._apply_stream_mlp(
			img, self.img,
			img_attn_in["mlp_shift"],
			img_attn_in["mlp_scale"],
			img_attn_in["mlp_gate"],
		)
		txt = self._apply_stream_mlp(
			txt, self.txt,
			txt_attn_in["mlp_shift"],
			txt_attn_in["mlp_scale"],
			txt_attn_in["mlp_gate"],
		)
		if self.use_geo_stream:
			geo = self._apply_stream_mlp(
				geo, self.geo,
				geo_attn_in["mlp_shift"],
				geo_attn_in["mlp_scale"],
				geo_attn_in["mlp_gate"],
			)
		return img, geo, txt


class MMSingleStreamBlock(nn.Module):
	"""Hunyuan single-stream block with MRoPE applied to every modality span."""

	def __init__(self, base_block: nn.Module, hy_config: DeepWorldHYModelConfig):
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
		self.kv_gates = _build_kv_gates(
			hy_config, next(base_block.parameters()).dtype
		)

	def forward(
		self,
		x: Tensor,
		vec: Tensor,
		img_len: int,
		geo_len: int,
		rope_spans: list[Span] | None,
		gate_spans: list[Span] | None,
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

		query, key = _apply_qk_rope(query, key, rope_spans)
		key, value = _apply_kv_gate(key, value, gate_spans, self.kv_gates)

		prefix_len = img_len + geo_len
		prefix_q, txt_q = query[:, :prefix_len], query[:, prefix_len:]
		prefix_k, txt_k = key[:, :prefix_len], key[:, prefix_len:]
		prefix_v, txt_v = value[:, :prefix_len], value[:, prefix_len:]

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

		self._install_triple_stream_blocks(self.hy_config)
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
		self.txt_in = self.transformer.txt_in

		trainable_dtype = next(self.transformer.parameters()).dtype
		
		if self.hy_config.use_geo_tokens:
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
			geo_hidden_size = self.geo_encoder.aggregator.frame_blocks[0].norm1.weight.size(0) * 2
			self.geo_in = nn.Sequential(
				nn.LayerNorm(geo_hidden_size, dtype=trainable_dtype),
				nn.Linear(geo_hidden_size, self.transformer.hidden_size, dtype=trainable_dtype),
				nn.GELU(),
				nn.Linear(self.transformer.hidden_size, self.transformer.hidden_size, dtype=trainable_dtype),
				nn.LayerNorm(self.transformer.hidden_size, dtype=trainable_dtype),
			)

		if self.hy_config.use_vis_tokens:
			self.vis_encoder = HunyuanVideo_1_5_Pipeline._load_vision_encoder(
				self.hy_config.checkpoint_path,
				device=None,
			)
			self.vis_encoder.requires_grad_(False)
			self.vis_in = getattr(self.transformer, "vision_in", None)
			if self.vis_in is None:
				vision_hidden_size = int(self.vis_encoder.model.config.hidden_size)
				self.vis_in = nn.Sequential(
					nn.LayerNorm(vision_hidden_size, dtype=trainable_dtype),
					nn.Linear(vision_hidden_size, self.transformer.hidden_size, dtype=trainable_dtype),
					nn.GELU(),
					nn.Linear(self.transformer.hidden_size, self.transformer.hidden_size, dtype=trainable_dtype),
					nn.LayerNorm(self.transformer.hidden_size, dtype=trainable_dtype),
				)

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
		if self.hy_config.use_vis_tokens:
			self.vis_encoder.eval()
		if self.hy_config.use_geo_tokens:
			self.geo_encoder.eval()

	def train(self, mode: bool = True):
		"""Set train mode while preserving eval mode for frozen encoders."""

		super().train(mode)
		self._set_frozen_modules_eval()
		return self

	def _install_triple_stream_blocks(self, hy_config: DeepWorldHYModelConfig) -> None:
		"""Replace Hunyuan blocks with MRoPE-aware wrapper blocks."""

		self.transformer.double_blocks = nn.ModuleList([
			MMTripleStreamBlock(block, hy_config=hy_config)
			for block in self.transformer.double_blocks
		])
		self.transformer.single_blocks = nn.ModuleList([
			MMSingleStreamBlock(block, hy_config=hy_config)
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
		rope_spans: list[Span],
		gate_spans: list[Span],
		stream_name: str,
	) -> tuple[Tensor, list[Span], list[Span]]:
		"""Shard one prefix stream along sequence length for Hunyuan sequence parallelism."""

		def localize_spans(
			spans: list[Span],
			local_start: int,
			local_end: int,
			slice_attachment,
		) -> list[Span]:
			"""Slice global spans down to a local sequence-parallel window."""

			local_spans: list[Span] = []
			for span in spans:
				span_end = span.start + span.length
				overlap_start = max(span.start, local_start)
				overlap_end = min(span_end, local_end)
				if overlap_start >= overlap_end:
					continue
				local_spans.append(Span(
					start=overlap_start - local_start,
					length=overlap_end - overlap_start,
					attachment=slice_attachment(span, overlap_start, overlap_end),
				))
			return local_spans

		def slice_rope_attachment(span: Span, overlap_start: int, overlap_end: int) -> tuple[Tensor, Tensor]:
			"""Slice RoPE frequencies to match a localized overlap."""

			freq_start = overlap_start - span.start
			freq_end = overlap_end - span.start
			freqs_cis = span.attachment
			return freqs_cis[0][freq_start:freq_end], freqs_cis[1][freq_start:freq_end]

		parallel_dims = get_parallel_state()
		if not getattr(parallel_dims, "sp_enabled", False):
			return tokens, rope_spans, gate_spans

		sp_size = parallel_dims.sp
		sp_rank = parallel_dims.sp_rank
		token_count = tokens.size(1)
		if token_count == 0:
			return tokens, [], []
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

		return (
			token_chunks[sp_rank],
			localize_spans(rope_spans, local_start, local_end, slice_rope_attachment),
			localize_spans(gate_spans, local_start, local_end, lambda span, *_: span.attachment),
		)

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
	
	def _uses_concat_condition(self) -> bool:
		"""Return whether the loaded Hunyuan patch embedder expects condition channels."""

		return bool(getattr(self.transformer.config, "concat_condition", True))

	def _video_patch_input(self, latents: Tensor) -> Tensor:
		"""Build empty i2v condition channels for target video tokens."""

		if not self._uses_concat_condition():
			return latents
		mask = torch.zeros(
			latents.size(0), 1, *latents.shape[-3:],
			device=latents.device, dtype=latents.dtype,
		)
		return torch.cat([latents, torch.zeros_like(latents), mask], dim=1)

	def _reference_patch_input(self, latents: Tensor) -> Tensor:
		"""Build i2v-style condition channels for reference image latents."""

		if not self._uses_concat_condition():
			return latents
		mask = torch.ones(
			latents.size(0), 1, *latents.shape[-3:],
			device=latents.device,
			dtype=latents.dtype,
		)
		return torch.cat([torch.zeros_like(latents), latents, mask], dim=1)

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

	def _encode_vis_refs(self, vis_ref_images: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
		"""Encode reference images with frozen SigLIP and project them to Hunyuan hidden size."""

		if not self.hy_config.use_vis_tokens:
			empty = torch.empty(1, 0, self.transformer.hidden_size, device=vis_ref_images.device)
			return empty, (1, 0, 0)

		if vis_ref_images.numel() == 0:
			empty = torch.empty(1, 0, self.transformer.hidden_size, device=vis_ref_images.device)
			return empty, (1, 0, 0)
		images_np = (vis_ref_images.detach().float().cpu().permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype("uint8")
		with torch.no_grad():
			vision_states = self.vis_encoder.encode_images(images_np).last_hidden_state
		projector_dtype = next(self.vis_in.parameters()).dtype
		vis_tokens = self.vis_in(vision_states.to(device=vis_ref_images.device, dtype=projector_dtype))
		vis_tokens = vis_tokens.view(1, -1, vis_tokens.size(-1))
		patch_size = int(self.vis_encoder.model.config.patch_size)
		# TODO: This is a bit messy here, `_encode_geo_refs` and `_encode_vae_refs` both handle grid size gracefully. Can you do better with `_encode_vis_refs` to achieve osnsistent coding style as well?
		image_size = int(self.vis_encoder.model.config.image_size)
		token_count = int(vision_states.size(1))
		grid_size = image_size // patch_size
		if grid_size * grid_size != token_count:
			grid_size = math.isqrt(token_count)
			if grid_size * grid_size != token_count:
				raise ValueError(f"Cannot infer a square SigLIP token grid from {token_count} visual tokens.")
		return vis_tokens, (1, grid_size, grid_size)

	def _encode_geo_refs(self, geo_ref_images: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
		"""Encode reference images with frozen VGGT and project geometry tokens."""

		if not self.hy_config.use_geo_tokens:
			empty = torch.empty(1, 0, self.transformer.hidden_size, device=geo_ref_images.device)
			return empty, (1, 0, 0)

		if geo_ref_images.numel() == 0:
			empty = torch.empty(1, 0, self.transformer.hidden_size, device=geo_ref_images.device)
			return empty, (1, 0, 0)

		device = geo_ref_images.device
		vggt_dtype = next(self.geo_encoder.aggregator.parameters()).dtype
		with torch.no_grad():
			aggregated_tokens, patch_start_idx = self.geo_encoder.aggregator(geo_ref_images.unsqueeze(0).to(dtype=vggt_dtype))
			geo_tokens = aggregated_tokens[-1][0, :, patch_start_idx:, :]
		projector_dtype = next(self.geo_in.parameters()).dtype
		geo_tokens = self.geo_in(geo_tokens.to(dtype=projector_dtype))
		geo_tokens = geo_tokens.view(1, -1, geo_tokens.size(-1))

		patch_size = self.geo_encoder.aggregator.patch_size
		grid_h = geo_ref_images.size(-2) // patch_size
		grid_w = geo_ref_images.size(-1) // patch_size
		return geo_tokens.to(device=device), (1, grid_h, grid_w)

	def _encode_vae_refs(self, vae_ref_images: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
		"""Encode reference images through the frozen VAE and Hunyuan image patch embedder."""

		if not self.hy_config.use_vae_tokens:
			empty = torch.empty(1, 0, self.transformer.hidden_size, device=vae_ref_images.device)
			return empty, (1, 0, 0)

		if vae_ref_images.numel() == 0:
			empty = torch.empty(1, 0, self.transformer.hidden_size, device=vae_ref_images.device)
			return empty, (1, 0, 0)

		ref_latents = self.encode_vae(vae_ref_images, sample_posterior=False)
		ref_input = self._reference_patch_input(ref_latents)
		ref_tokens = self.transformer.img_in(ref_input.to(dtype=next(self.transformer.parameters()).dtype))
		ref_tokens = ref_tokens.view(1, -1, ref_tokens.size(-1))
		return ref_tokens, self._latent_patch_grid(ref_latents)

	def _build_rope_and_gate_spans(
		self,
		video_grid: tuple[int, int, int],
		video_token_count: int,
		vae_grid: tuple[int, int, int],
		vae_token_count: int,
		geo_grid: tuple[int, int, int],
		geo_token_count: int,
		vis_grid: tuple[int, int, int],
		vis_token_count: int,
		txt_token_count: int,
		reference_count: int,
		device: torch.device,
	) -> tuple[list[Span], list[Span], list[Span], list[Span], list[Span], list[Span]]:
		"""Build RoPE and K/V-gate spans in video, VAE, geometry, visual, text order."""

		cursor = 0
		img_rope_spans: list[Span] = []
		geo_rope_spans: list[Span] = []
		txt_rope_spans: list[Span] = []

		def add_rope_span(spans: list[Span], start: int, positions: Tensor) -> None:
			"""Append one span's RoPE frequencies when the span is non-empty."""

			if positions.numel() == 0:
				return
			freqs_cis = self._build_mrope_freqs(positions)
			if freqs_cis is not None:
				spans.append(Span(start, positions.size(0), freqs_cis))

		positions, cursor = self._make_grid_positions(video_grid, cursor, device)
		add_rope_span(img_rope_spans, 0, positions)

		if self.hy_config.use_mrope:
			vae_tokens_per_ref = math.prod(vae_grid) if vae_token_count > 0 else 0
			vae_start = video_token_count
			for _ in range(reference_count if vae_tokens_per_ref > 0 else 0):
				positions, cursor = self._make_grid_positions(vae_grid, cursor, device)
				add_rope_span(img_rope_spans, vae_start, positions)
				vae_start += positions.size(0)

			geo_tokens_per_ref = math.prod(geo_grid) if geo_token_count > 0 else 0
			geo_start = 0
			for _ in range(reference_count if geo_tokens_per_ref > 0 else 0):
				positions, cursor = self._make_grid_positions(geo_grid, cursor, device)
				add_rope_span(geo_rope_spans, geo_start, positions)
				geo_start += positions.size(0)

			vis_tokens_per_ref = math.prod(vis_grid) if vis_token_count > 0 else 0
			vis_start = 0
			for _ in range(reference_count if vis_tokens_per_ref > 0 else 0):
				positions, cursor = self._make_grid_positions(vis_grid, cursor, device)
				add_rope_span(txt_rope_spans, vis_start, positions)
				vis_start += positions.size(0)
			
			if txt_token_count > 0:
				positions, cursor = self._make_text_positions(txt_token_count, cursor, device)
				add_rope_span(txt_rope_spans, vis_token_count, positions)

		img_gate_spans: list[Span] = []
		geo_gate_spans: list[Span] = []
		txt_gate_spans: list[Span] = []
		if vae_token_count > 0:
			img_gate_spans.append(Span(video_token_count, vae_token_count, "vae"))
		if geo_token_count > 0:
			geo_gate_spans.append(Span(0, geo_token_count, "geo"))
		if vis_token_count > 0:
			txt_gate_spans.append(Span(0, vis_token_count, "vis"))
		if txt_token_count > 0:
			txt_gate_spans.append(Span(vis_token_count, txt_token_count, "txt"))
		return img_rope_spans, geo_rope_spans, txt_rope_spans, img_gate_spans, geo_gate_spans, txt_gate_spans

	def _sample_drop(self, prob: float, device: torch.device) -> bool:
		"""Sample one Bernoulli token-drop decision."""

		return bool(prob > 0.0 and torch.rand((), device=device) < prob)

	def _build_condition_plan(
		self,
		batch: dict[str, Any],
		device: torch.device,
		force_drop_condition: bool = False,
		sample_dropout: bool = True,
	) -> dict[str, bool]:
		"""Resolve which condition-token encoders should run for this sample."""

		is_pure_t2v = int(batch["vis_ref_images"].size(0)) == 0
		use_vae = self.hy_config.use_vae_tokens and not is_pure_t2v
		use_geo = self.hy_config.use_geo_tokens and not is_pure_t2v
		use_vis = self.hy_config.use_vis_tokens and not is_pure_t2v
		use_txt = True
		if force_drop_condition:
			return {"vae": False, "geo": False, "vis": False, "txt": False}

		should_sample_dropout = self.training and sample_dropout
		if should_sample_dropout and self._sample_drop(self.hy_config.condition_dropout_prob, device):
			return {"vae": False, "geo": False, "vis": False, "txt": False}
		if should_sample_dropout and not is_pure_t2v:
			use_vae = use_vae and not self._sample_drop(self.hy_config.drop_vae_tokens_prob, device)
			use_geo = use_geo and not self._sample_drop(self.hy_config.drop_geo_tokens_prob, device)
			use_vis = use_vis and not self._sample_drop(self.hy_config.drop_vis_tokens_prob, device)
			use_txt = use_txt and not self._sample_drop(self.hy_config.drop_txt_tokens_prob, device)
		return {"vae": use_vae, "geo": use_geo, "vis": use_vis, "txt": use_txt}

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
		force_drop_condition: bool = False,
		sample_dropout: bool = True,
	) -> Tensor:
		"""Run the Hunyuan three-stream transformer and predict flow velocity."""

		device = latents_noised.device
		dtype = next(self.transformer.parameters()).dtype
		hidden_size = self.transformer.hidden_size
		condition_plan = self._build_condition_plan(
			batch,
			device=device,
			force_drop_condition=force_drop_condition,
			sample_dropout=sample_dropout,
		)
		video_input = self._video_patch_input(latents_noised)
		video_grid = self._latent_patch_grid(latents_noised)
		video_token_count = math.prod(video_grid)

		img = self.transformer.img_in(video_input.to(dtype=dtype))
		img = img + self.modality_embeddings["video"].to(device=device, dtype=img.dtype)

		vae_tokens = torch.empty(1, 0, hidden_size, device=device, dtype=dtype)
		vae_grid = (1, 0, 0)
		geo_tokens = torch.empty(1, 0, hidden_size, device=device, dtype=dtype)
		geo_grid = (1, 0, 0)
		vis_tokens = torch.empty(1, 0, hidden_size, device=device, dtype=dtype)
		vis_mask = torch.empty(1, 0, device=device, dtype=torch.long)
		vis_grid = (1, 0, 0)
		txt_tokens = torch.empty(1, 0, hidden_size, device=device, dtype=dtype)
		txt_mask = torch.empty(1, 0, device=device, dtype=torch.long)
		
		if condition_plan["vae"]:
			refs = batch["vae_ref_images"].to(device=device, non_blocking=True)
			vae_tokens, vae_grid = self._encode_vae_refs(refs)
			vae_tokens = vae_tokens + self.modality_embeddings["vae"].to(device=device, dtype=img.dtype)
		
		if condition_plan["geo"]:
			refs = batch["geo_ref_images"].to(device=device, non_blocking=True)
			geo_tokens, geo_grid = self._encode_geo_refs(refs)
			geo_tokens = geo_tokens + self.modality_embeddings["geo"].to(device=device, dtype=img.dtype)
		
		if condition_plan["vis"]:
			refs = batch["vis_ref_images"].to(device=device, non_blocking=True)
			vis_tokens, vis_grid = self._encode_vis_refs(refs)
			vis_tokens = vis_tokens + self.modality_embeddings["vis"].to(device=device, dtype=img.dtype)
			vis_mask = torch.ones(vis_tokens.size()[:2], device=device, dtype=torch.long)

		if condition_plan["txt"]:
			text_states, txt_mask = self._encode_text(batch["prompt"], device=device, dtype=dtype)
			if self.transformer.text_projection == "linear":
				txt_tokens = self.txt_in(text_states)
			else:
				txt_tokens = self.txt_in(text_states, timesteps, txt_mask if self.transformer.use_attention_mask else None)
			txt_tokens = txt_tokens + self.modality_embeddings["txt"].to(device=device, dtype=img.dtype)
		
		img = torch.cat([img, vae_tokens.to(dtype=img.dtype)], dim=1)
		txt = torch.cat([vis_tokens.to(dtype=img.dtype), txt_tokens.to(dtype=img.dtype)], dim=1)
		txt_mask = torch.cat([vis_mask, txt_mask], dim=1)
		geo = geo_tokens.to(dtype=img.dtype)

		reference_count = int(batch["vis_ref_images"].size(0))
		(
			img_rope_spans,
			geo_rope_spans,
			txt_rope_spans,
			img_gate_spans,
			geo_gate_spans,
			txt_gate_spans,
		) = self._build_rope_and_gate_spans(
			video_grid=video_grid,
			video_token_count=video_token_count,
			vae_grid=vae_grid,
			vae_token_count=int(vae_tokens.size(1)),
			geo_grid=geo_grid,
			geo_token_count=int(geo_tokens.size(1)),
			vis_grid=vis_grid,
			vis_token_count=int(vis_tokens.size(1)),
			txt_token_count=int(txt_tokens.size(1)),
			reference_count=reference_count,
			device=device,
		)
		img, img_rope_spans, img_gate_spans = self._split_for_sequence_parallel(
			img, img_rope_spans, img_gate_spans, "image",
		)
		geo, geo_rope_spans, geo_gate_spans = self._split_for_sequence_parallel(
			geo, geo_rope_spans, geo_gate_spans, "geometry",
		)

		self.transformer.attn_param["thw"] = list(video_grid)
		vec = self._time_vector(timesteps, dtype=img.dtype)
		
		for index, block in enumerate(self.transformer.double_blocks):
			self.transformer.attn_param["layer-name"] = f"world_double_block_{index + 1}"
			img, geo, txt = block(
				img=img,
				geo=geo,
				txt=txt,
				vec=vec,
				img_rope_spans=img_rope_spans,
				geo_rope_spans=geo_rope_spans,
				txt_rope_spans=txt_rope_spans,
				img_gate_spans=img_gate_spans,
				geo_gate_spans=geo_gate_spans,
				txt_gate_spans=txt_gate_spans,
				text_mask=txt_mask,
				attn_param=self.transformer.attn_param,
				is_flash=False,
				block_idx=index,
			)

		img_len, geo_len = img.size(1), geo.size(1)
		x = torch.cat([img, geo, txt], dim=1)
		single_rope_spans = (
			img_rope_spans
			+ [span.shifted(img_len) for span in geo_rope_spans]
			+ [span.shifted(img_len + geo_len) for span in txt_rope_spans]
		)
		single_gate_spans = (
			img_gate_spans
			+ [span.shifted(img_len) for span in geo_gate_spans]
			+ [span.shifted(img_len + geo_len) for span in txt_gate_spans]
		)
		
		for index, block in enumerate(self.transformer.single_blocks):
			self.transformer.attn_param["layer-name"] = f"world_single_block_{index + 1}"
			x = block(
				x=x,
				vec=vec,
				img_len=img_len,
				geo_len=geo_len,
				rope_spans=single_rope_spans,
				gate_spans=single_gate_spans,
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
			shift=self.hy_config.eval_timestep_shift,
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
				sample_dropout=False,
			)
			if self.hy_config.cfg_guidance_scale > 1.0:
				model_output_uncond = self.predict_velocity(
					model_input,
					timestep.view(1),
					batch,
					force_drop_condition=True,
					sample_dropout=False,
				)
				model_output_text = model_output
				model_output = model_output_uncond + self.hy_config.cfg_guidance_scale * (
					model_output_text - model_output_uncond
				)
				if self.hy_config.cfg_guidance_rescale > 0.0:
					model_output = rescale_noise_cfg(
						model_output,
						model_output_text,
						guidance_rescale=self.hy_config.cfg_guidance_rescale,
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
		latents = self.encode_vae(video, sample_posterior=self.hy_config.video_vae_sample)

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
