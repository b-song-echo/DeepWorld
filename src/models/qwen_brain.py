from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import GradientCheckpointingLayer

from vggt.models.vggt import VGGT
from src.config import QwenBrainConfig
from src.utils.compat import resolve_torch_dtype


VIS_MODALITY = 0
GEO_MODALITY = 1
TXT_MODALITY = 2
GEN_MODALITY = 3


class LoraLinear(nn.Module):
	"""Wrap a frozen linear layer with a trainable low-rank residual branch.

	Args:
		base_layer: Frozen base `nn.Linear` layer.
		rank: LoRA rank.
		alpha: LoRA scaling numerator.
		dropout: Dropout probability applied before the low-rank up projection.
	"""

	def __init__(
		self, base_layer: nn.Linear,
		rank: int, alpha: int, dropout: float
	):
		super().__init__()
		self.base_layer = base_layer
		self.base_layer.requires_grad_(False)

		self.rank = rank
		self.scaling = alpha / rank
		self.dropout = nn.Dropout(dropout)
		device = base_layer.weight.device
		dtype = base_layer.weight.dtype

		self.lora_a = nn.Linear(
			base_layer.in_features, rank, bias=False,
			device=device, dtype=dtype,
		)
		self.lora_b = nn.Linear(
			rank, base_layer.out_features, bias=False,
			device=device, dtype=dtype,
		)

		nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
		nn.init.zeros_(self.lora_b.weight)

	def forward(self, hidden_states: Tensor) -> Tensor:
		"""Apply the frozen linear layer plus the trainable LoRA residual.

		Args:
			hidden_states: Input tensor whose last dimension matches the wrapped layer.

		Returns:
			The adapted linear output.
		"""

		middle = self.dropout(self.lora_a(hidden_states))
		residual = self.lora_b(middle) * self.scaling
		return self.base_layer(hidden_states) + residual


def inject_lora_layers(
	module: nn.Module, target_names: Iterable[str],
	rank: int, alpha: int, dropout: float
) -> None:
	"""Recursively replace selected linear submodules with `LoRALinear`.

	Args:
		module: Root module to traverse.
		target_names: Child-module names that should receive LoRA wrappers.
		rank: LoRA rank.
		alpha: LoRA scaling numerator.
		dropout: LoRA dropout probability.
	"""

	for name, child in list(module.named_children()):
		if isinstance(child, nn.Linear) and name in target_names:
			lora_child = LoraLinear(child, rank=rank, alpha=alpha, dropout=dropout)
			setattr(module, name, lora_child)
		else:
			inject_lora_layers(child, target_names, rank, alpha, dropout)


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


class QwenBrain(nn.Module):
	"""Build multimodal Qwen inputs and extract Wan-aligned generation states.

	The module combines:

	- frozen Qwen visual features,
	- frozen VGGT tokens projected into Qwen hidden space,
	- learned modality marker tokens between the multimodal spans,
	- prompt token embeddings,
	- learned generation probe tokens whose final hidden states condition Wan.

	Args:
		qwen_config: Qwen brain branch configuration.
		gradient_checkpointing: Whether to enable checkpointing in the Qwen text stack.
	"""

	def __init__(self, qwen_config: QwenBrainConfig, gradient_checkpointing: bool = True):
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
		image_grid_thw: Tensor, vis_ref_counts: Tensor,
	) -> tuple[list[list[Tensor]], list[list[tuple[int, int, int]]]]:
		"""Extract visual tokens from Qwen's frozen vision encoder.

		Args:
			pixel_values: Flattened Qwen image inputs for all references in the batch.
			image_grid_thw: Qwen feature-grid metadata for each flattened reference.
			vis_ref_counts: Number of references per batch element.

		Returns:
			A tuple containing:
			- visual feature tensors regrouped per sample,
			- the corresponding `(t, h, w)` grids after Qwen's spatial merge.
		"""

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
			if feature_lengths:
				flat_features = torch.cat(image_features, dim=0).to(bridge_dtype)
				flat_features = self.vis_bridge(flat_features)
				image_features = torch.split(flat_features, feature_lengths)
		else:
			image_features = image_features.to(bridge_dtype)
			image_features = self.vis_bridge(image_features)

		vis_groups: list[list[Tensor]] = []
		vis_grids: list[list[tuple[int, int, int]]] = []
		offset = 0
		for count in vis_ref_counts.tolist():
			vis_groups.append(list(image_features[offset:(offset + count)]))
			sample_grids: list[tuple[int, int, int]] = []
			for grid in image_grid_thw[offset:(offset + count)]:
				grid_t, grid_h, grid_w = grid.tolist()
				patch_size = self.vis_patch_size
				grid_h = grid_h // patch_size
				grid_w = grid_w // patch_size
				sample_grids.append((grid_t, grid_h, grid_w))
			vis_grids.append(sample_grids)
			offset += count
		return vis_groups, vis_grids

	def _encode_geometry(
		self, geo_images: Tensor,
		reference_mask: Tensor,
	) -> tuple[list[list[Tensor]], list[list[tuple[int, int, int]]]]:
		"""Extract geometry tokens from the frozen VGGT aggregator.

		Args:
			geo_images: Shared reference-image tensor with shape `(B, S_ref, 3, H, W)`.
			reference_mask: Boolean mask identifying valid reference-image slots.

		Returns:
			A tuple containing:
			- geometry feature tensors regrouped per sample,
			- the corresponding `(t, h, w)` token grids.
		"""

		device = next(self.parameters()).device
		vggt_dtype = next(self.geometry_encoder.aggregator.parameters()).dtype
		bridge_dtype = self.geo_bridge[0].weight.dtype
		with torch.no_grad():
			aggregated_tokens, patch_start_idx = self.geometry_encoder.aggregator(
				geo_images.to(device, vggt_dtype, non_blocking=True)
			)
			geo_tokens = aggregated_tokens[-1][:, :, patch_start_idx:, :]
		geo_tokens = self.geo_bridge(geo_tokens.to(bridge_dtype))

		patch_size = self.geometry_encoder.aggregator.patch_size
		grid_h = geo_images.size(-2) // patch_size
		grid_w = geo_images.size(-1) // patch_size

		geo_groups: list[list[Tensor]] = []
		geo_grids: list[list[tuple[int, int, int]]] = []
		for batch_index in range(geo_images.size(0)):
			sample_features: list[Tensor] = []
			sample_grids: list[tuple[int, int, int]] = []
			for ref_index in range(geo_images.size(1)):
				if not reference_mask[batch_index, ref_index]:
					continue
				sample_features.append(geo_tokens[batch_index, ref_index])
				sample_grids.append((1, grid_h, grid_w))
			geo_groups.append(sample_features)
			geo_grids.append(sample_grids)
		return geo_groups, geo_grids

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
		vis_groups: list[list[Tensor]],
		vis_grids: list[list[tuple[int, int, int]]],
		geo_groups: list[list[Tensor]],
		geo_grids: list[list[tuple[int, int, int]]],
		txt_input_ids: Tensor,
		txt_attention_mask: Tensor,
		latent_patch_grids: Tensor,
	) -> dict[str, Tensor | list[tuple[int, int]]]:
		"""Assemble the direct-embedding Qwen language-model inputs.

		Args:
			vis_groups: Per-sample visual feature sequences.
			vis_grids: Per-sample visual token grids.
			geo_groups: Per-sample geometry feature sequences.
			geo_grids: Per-sample geometry token grids.
			txt_input_ids: Tokenized prompt IDs.
			txt_attention_mask: Prompt attention masks.
			latent_patch_grids: Target Wan latent token grids, one per batch element.

		Returns:
			A dictionary containing padded embeddings, masks, position IDs, modality IDs,
			and bookkeeping for where the generation-token slice lives in each sample.
			Each sample sequence is prefixed with learned modality marker tokens for the
			visual, geometry, text, and generation spans.
		"""

		device = next(self.parameters()).device
		txt_embeddings = self.language_model.embed_tokens(
			txt_input_ids.to(device=device, non_blocking=True)
		)
		txt_embeddings = self.txt_bridge(txt_embeddings)
		txt_attention_mask = txt_attention_mask.to(device, non_blocking=True).bool()
		batch_size = txt_input_ids.size(0)

		sequences: list[Tensor] = []
		position_ids: list[Tensor] = []
		modality_ids: list[Tensor] = []
		gen_lengths: list[int] = []

		for batch_index in range(batch_size):
			parts: list[Tensor] = []
			part_positions: list[Tensor] = []
			part_modalities: list[Tensor] = []
			cursor = 0

			cursor = self._append_segment_token(
				"vis", VIS_MODALITY, parts, part_positions,
				part_modalities, cursor, device, txt_embeddings.dtype,
			)

			for features, grid in zip(vis_groups[batch_index], vis_grids[batch_index]):
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

			for features, grid in zip(geo_groups[batch_index], geo_grids[batch_index]):
				features = features.to(device)
				positions, cursor = self._make_grid_positions(grid, cursor, device)
				self._append_span(
					parts, part_positions, part_modalities,
					features, positions, GEO_MODALITY,
				)

			txt_features = txt_embeddings[batch_index, txt_attention_mask[batch_index]]
			cursor = self._append_segment_token(
				"txt", TXT_MODALITY, parts, part_positions,
				part_modalities, cursor, device, txt_embeddings.dtype,
			)
			positions, cursor = self._make_txt_positions(txt_features.size(0), cursor, device)
			self._append_span(
				parts, part_positions, part_modalities,
				txt_features, positions, TXT_MODALITY,
			)

			latent_grid = tuple(int(value) for value in latent_patch_grids[batch_index].tolist())
			cursor = self._append_segment_token(
				"gen", GEN_MODALITY, parts, part_positions,
				part_modalities, cursor, device, txt_embeddings.dtype,
			)
			gen_positions, cursor = self._make_grid_positions(latent_grid, cursor, device)
			gen_length = int(torch.prod(latent_patch_grids[batch_index]).item())
			gen_lengths.append(gen_length)
			gen_tokens = self.gen_slot_token.to(device=device, dtype=txt_embeddings.dtype).expand(gen_length, -1)
			self._append_span(
				parts, part_positions, part_modalities,
				gen_tokens, gen_positions, GEN_MODALITY,
			)

			sequences.append(torch.cat(parts, dim=0))
			position_ids.append(torch.cat(part_positions, dim=1))
			modality_ids.append(torch.cat(part_modalities, dim=0))

		max_sequence_length = max(sequence.size(0) for sequence in sequences)
		inputs_embeds = torch.zeros(batch_size, max_sequence_length, self.hidden_size, device=device, dtype=txt_embeddings.dtype)
		attention_mask = torch.zeros(batch_size, max_sequence_length, device=device, dtype=torch.long)
		batched_position_ids = torch.zeros(3, batch_size, max_sequence_length, device=device, dtype=torch.long)
		batched_modality_ids = torch.full((batch_size, max_sequence_length), TXT_MODALITY, device=device, dtype=torch.long)
		gen_mask = torch.zeros(batch_size, max(gen_lengths), device=device, dtype=torch.bool)

		gen_slices: list[tuple[int, int]] = []
		for batch_index, (sequence, pos, mods, gen_length) in enumerate(zip(sequences, position_ids, modality_ids, gen_lengths)):
			sequence_length = sequence.size(0)
			inputs_embeds[batch_index, :sequence_length] = sequence
			attention_mask[batch_index, :sequence_length] = 1
			batched_position_ids[:, batch_index, :sequence_length] = pos
			batched_modality_ids[batch_index, :sequence_length] = mods
			gen_start = sequence_length - gen_length
			gen_slices.append((gen_start, sequence_length))
			gen_mask[batch_index, :gen_length] = True

		return {
			"inputs_embeds": inputs_embeds,
			"attention_mask": attention_mask,
			"position_ids": batched_position_ids,
			"modality_ids": batched_modality_ids,
			"gen_mask": gen_mask,
			"gen_slices": gen_slices,
		}

	def forward(self, batch: dict[str, Tensor], latent_patch_grids: Tensor) -> dict[str, Tensor]:
		"""Run the full Qwen/VGGT conditioning pipeline.

		Args:
			batch: Training batch produced by `WorldModelBatchCollator`.
			latent_patch_grids: Desired Wan latent token grids for each batch element.

		Returns:
			A dictionary containing:
			- `gen_hidden_states`: final hidden states of the learned generation tokens,
			- `gen_mask`: valid-token mask for those generation slots.
		"""

		vis_groups, vis_grids = self._encode_visual(
			batch["qwen_vis_pixel_values"],
			batch["qwen_vis_grid_thw"],
			batch["vis_ref_counts"],
		)
		geo_groups, geo_grids = self._encode_geometry(batch["geo_images"], batch["reference_mask"])
		language_inputs = self._build_language_inputs(
			vis_groups=vis_groups,
			vis_grids=vis_grids,
			geo_groups=geo_groups,
			geo_grids=geo_grids,
			txt_input_ids=batch["txt_input_ids"],
			txt_attention_mask=batch["txt_attention_mask"],
			latent_patch_grids=latent_patch_grids,
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
		max_gen_length = language_inputs["gen_mask"].size(1)
		gen_hidden_states = torch.zeros(
			last_hidden_state.size(0),
			max_gen_length,
			last_hidden_state.size(-1),
			device=last_hidden_state.device,
			dtype=last_hidden_state.dtype,
		)

		for batch_index, (start, end) in enumerate(language_inputs["gen_slices"]):
			length = end - start
			gen_hidden_states[batch_index, :length] = last_hidden_state[batch_index, start:end]

		return {
			"gen_hidden_states": gen_hidden_states,
			"gen_mask": language_inputs["gen_mask"],
		}
