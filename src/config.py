from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar

import yaml


def resolve_lora_alpha(alpha: int | None, rank: int) -> int:
	"""Resolve a LoRA alpha value, defaulting to `2 * rank` when omitted."""

	return 2 * rank if alpha is None else int(alpha)


@dataclass
class DatasetConfig:
	"""Dataset and dataloader settings shared by all world models."""

	manifest_path: str = ""
	eval_manifest_path: str = ""
	data_root: str = ""
	num_samples: int = 0
	video_width: int = 448
	video_height: int = 448
	video_fps: float = 24.0
	video_duration: float = 5.0
	prompt_rich_prob: float = 0.5
	prompt_medium_prob: float = 0.3
	prompt_coarse_prob: float = 0.2
	mixing_t2v_prob: float = 0.2
	vis_image_size: int = 448
	geo_image_size: int | None = None
	max_text_length: int = 1024
	eval_num_samples: int = 0
	shuffle: bool = True
	num_workers: int = 8
	prefetch_factor: int | None = None
	pin_memory: bool = True
	persistent_workers: bool = True

	def __post_init__(self) -> None:
		"""Validate curated-manifest preprocessing settings."""

		if self.num_samples < 0:
			raise ValueError(f"`dataset.num_samples` must be non-negative, got {self.num_samples}.")
		if self.video_width <= 0 or self.video_height <= 0:
			raise ValueError(
				"`dataset.video_width` and `dataset.video_height` must be positive, "
				f"got {(self.video_width, self.video_height)}."
			)
		if self.video_fps <= 0:
			raise ValueError(f"`dataset.video_fps` must be positive, got {self.video_fps}.")
		if self.video_duration <= 0:
			raise ValueError(f"`dataset.video_duration` must be positive, got {self.video_duration}.")
		if self.vis_image_size <= 0:
			raise ValueError(f"`dataset.vis_image_size` must be positive, got {self.vis_image_size}.")
		if self.eval_num_samples < 0:
			raise ValueError(f"`dataset.eval_num_samples` must be non-negative, got {self.eval_num_samples}.")
		for name, value in (
			("prompt_rich_prob", self.prompt_rich_prob),
			("prompt_medium_prob", self.prompt_medium_prob),
			("prompt_coarse_prob", self.prompt_coarse_prob),
		):
			if value < 0:
				raise ValueError(f"`dataset.{name}` must be non-negative, got {value}.")
		if self.prompt_rich_prob + self.prompt_medium_prob + self.prompt_coarse_prob <= 0:
			raise ValueError("At least one prompt probability must be positive.")
		if not 0.0 <= self.mixing_t2v_prob <= 1.0:
			raise ValueError(
				"`dataset.mixing_t2v_prob` must be in [0, 1], "
				f"got {self.mixing_t2v_prob}."
			)

	@property
	def video_num_frames(self) -> int:
		"""Return the endpoint-inclusive frame count for the configured clip."""

		return int(round(self.video_fps * self.video_duration)) + 1


@dataclass
class TrainingConfig:
	"""Training orchestration shared by all world models."""

	output_dir: str = "outputs/world_model"
	seed: int = 42
	max_total_epochs: int | None = 10
	max_total_steps: int | None = None
	sequence_parallel_size: int = 1
	data_parallel_replicate: int = 1
	gradient_accumulation_steps: int = 1
	log_every: int = 10
	eval_every: int = 0
	save_every: int = 1000
	mixed_precision: str = "bf16"
	use_fsdp: bool = True
	gradient_checkpointing: bool = True
	pretrained_model_path: str | None = None
	
	def __post_init__(self) -> None:
		"""Validate bounded stopping and distributed orchestration settings."""

		if self.max_total_epochs is None and self.max_total_steps is None:
			raise ValueError("At least one of `training.max_total_epochs` or `training.max_total_steps` must be set.")
		if self.max_total_epochs is not None and self.max_total_epochs <= 0:
			raise ValueError(
				f"`training.max_total_epochs` must be positive when set, got {self.max_total_epochs}."
			)
		if self.max_total_steps is not None and self.max_total_steps <= 0:
			raise ValueError(
				f"`training.max_total_steps` must be positive when set, got {self.max_total_steps}."
			)
		if self.sequence_parallel_size < 1:
			raise ValueError(
				"`training.sequence_parallel_size` must be at least 1, "
				f"got {self.sequence_parallel_size}."
			)
		if self.data_parallel_replicate < 1 and self.data_parallel_replicate != -1:
			raise ValueError(
				"`training.data_parallel_replicate` must be positive or -1, "
				f"got {self.data_parallel_replicate}."
			)
		if self.gradient_accumulation_steps < 1:
			raise ValueError(
				"`training.gradient_accumulation_steps` must be at least 1, "
				f"got {self.gradient_accumulation_steps}."
			)
		if self.log_every < 0:
			raise ValueError(f"`training.log_every` must be non-negative, got {self.log_every}.")
		if self.eval_every < 0:
			raise ValueError(f"`training.eval_every` must be non-negative, got {self.eval_every}.")
		if self.save_every < 0:
			raise ValueError(f"`training.save_every` must be non-negative, got {self.save_every}.")


@dataclass
class OptimizerConfig:
	"""Optimizer and LR-schedule settings shared by all world models."""

	learning_rate: float = 2e-5
	lr_schedule: str = "cosine"
	weight_decay: float = 1e-2
	betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
	eps: float = 1e-8
	max_grad_norm: float = 1.0
	warmup_steps: int = 500

	def __post_init__(self) -> None:
		"""Validate optimizer schedule settings."""

		self.lr_schedule = self.lr_schedule.lower()
		if self.lr_schedule not in {"cosine", "constant"}:
			raise ValueError(f"`optimizer.lr_schedule` must be either `cosine` or `constant`, got {self.lr_schedule!r}.")


@dataclass
class DeepWorldQWOptimizerConfig(OptimizerConfig):
	"""Optimizer settings specific to the Qwen/Wan DeepWorld variant."""

	brain_language_model_learning_rate: float | None = None
	brain_others_learning_rate: float | None = None
	renderer_transformer_learning_rate: float | None = None
	renderer_others_learning_rate: float | None = None


@dataclass
class DeepWorldHYOptimizerConfig(OptimizerConfig):
	"""Optimizer settings specific to the Hunyuan DeepWorld variant."""

	adapter_learning_rate: float | None = None
	geo_stream_learning_rate: float | None = None
	projector_learning_rate: float | None = None


@dataclass
class DeepWorldQWBrainConfig:
	"""Configuration for the DeepWorldQW Qwen/VGGT conditioning branch."""

	checkpoint_path: str = "checkpoints/Qwen3-VL-8B-Instruct"
	transformer_dtype: str | None = "bfloat16"
	vggt_checkpoint_path: str = "checkpoints/VGGT-1B"
	vggt_dtype: str | None = None
	lora_rank: int = 32
	lora_alpha: int | None = None
	lora_dropout: float = 0.05
	lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
	routed_ffn_mode: str = "lora"
	routed_ffn_lora_rank: int = 32
	routed_ffn_lora_alpha: int | None = None
	routed_ffn_lora_dropout: float = 0.05

	def __post_init__(self) -> None:
		"""Resolve default LoRA scaling values."""

		self.lora_alpha = resolve_lora_alpha(self.lora_alpha, self.lora_rank)
		self.routed_ffn_lora_alpha = resolve_lora_alpha(self.routed_ffn_lora_alpha, self.routed_ffn_lora_rank)


@dataclass
class DeepWorldQWRendererConfig:
	"""Configuration for the DeepWorldQW Wan renderer branch."""

	checkpoint_path: str = "checkpoints/Wan2.2-TI2V-5B-Diffusers"
	transformer_dtype: str | None = "bfloat16"
	vae_dtype: str | None = None
	vae_enable_slicing: bool = False
	vae_enable_tiling: bool = False
	vae_sample_posterior: bool = False
	condition_injection_mode: str = "input_addition"
	condition_proj_init: str = "zero"
	condition_proj_init_std: float | None = None
	condition_dropout_prob: float = 0.1
	train_scheduler_steps: int | None = None
	inference_steps: int = 50

	def __post_init__(self) -> None:
		"""Validate Wan renderer training and initialization settings."""

		self.condition_injection_mode = self.condition_injection_mode.lower()
		if self.condition_injection_mode not in {"input_addition", "cross_attention"}:
			raise ValueError(
				"`renderer.condition_injection_mode` must be `input_addition` or `cross_attention`, "
				f"got {self.condition_injection_mode!r}."
			)
		self.condition_proj_init = self.condition_proj_init.lower()
		if self.condition_proj_init not in {"zero", "normal", "default"}:
			raise ValueError(
				"`renderer.condition_proj_init` must be `zero`, `normal`, or `default`, "
				f"got {self.condition_proj_init!r}."
			)
		if self.condition_proj_init == "normal" and self.condition_proj_init_std is None:
			self.condition_proj_init_std = 1e-3
		if self.condition_proj_init_std is not None and self.condition_proj_init_std <= 0:
			raise ValueError(
				"`renderer.condition_proj_init_std` must be positive, "
				f"got {self.condition_proj_init_std}."
			)
		if not 0.0 <= self.condition_dropout_prob <= 1.0:
			raise ValueError(
				"`renderer.condition_dropout_prob` must be in [0, 1], "
				f"got {self.condition_dropout_prob}."
			)


@dataclass
class DeepWorldHYModelConfig:
	"""Model architecture and checkpoint settings for DeepWorldHY."""

	checkpoint_path: str = "checkpoints/HunyuanVideo-1.5"
	transformer_version: str = "480p_i2v"
	transformer_dtype: str | None = "bfloat16"
	vae_dtype: str | None = "float16"
	vggt_checkpoint_path: str = "checkpoints/VGGT-1B"
	vggt_dtype: str | None = None
	attention_mode: str = "flash"
	condition_dropout_prob: float = 0.1
	drop_vae_tokens_prob: float = 0.0
	drop_vis_tokens_prob: float = 0.0
	drop_geo_tokens_prob: float = 0.0
	drop_txt_tokens_prob: float = 0.0
	use_vae_tokens: bool = True
	use_vis_tokens: bool = True
	use_geo_tokens: bool = True
	use_mrope: bool = True
	video_vae_sample: bool = False
	cfg_guidance_scale: float = 1.0
	cfg_guidance_rescale: float = 0.0
	# TODO: Merge `use_xxx_kv_gate` and `xxx_kv_gate_init` into a single one: `xxx_kv_gate`. When set to None, do not use gates, otherwise, use it as init value for learnable gates.
	use_vae_kv_gate: bool = True
	use_geo_kv_gate: bool = True
	use_vis_kv_gate: bool = True
	use_txt_kv_gate: bool = False
	vae_kv_gate_init: float = 0.0
	geo_kv_gate_init: float = 0.0
	vis_kv_gate_init: float = 0.0
	txt_kv_gate_init: float = 0.0
	lora_rank: int = 16
	lora_alpha: int | None = None
	lora_dropout: float = 0.05
	lora_target_modules: List[str] = field(default_factory=lambda: [
		"img_attn_q", "img_attn_k", "img_attn_v", "img_attn_proj",
		"txt_attn_q", "txt_attn_k", "txt_attn_v", "txt_attn_proj",
		"linear1_q", "linear1_k", "linear1_v",
	])
	geo_stream_init: str = "copy_image"
	guidance: float = 6016.0
	num_train_timesteps: int = 1000
	train_timestep_shift: float = 3.0
	# TODO: You included this config but never used it. I have told you, never leave unused or redundant code, either use it, or remove it.
	eval_timestep_shift: float = 5.0
	snr_type: str = "lognorm"
	inference_steps: int = 50

	def __post_init__(self) -> None:
		"""Validate DeepWorldHY model settings."""

		self.lora_alpha = resolve_lora_alpha(self.lora_alpha, self.lora_rank)
		self.attention_mode = self.attention_mode.lower()
		self.geo_stream_init = self.geo_stream_init.lower()
		self.snr_type = self.snr_type.lower()
		if self.attention_mode == "flex-block-attn":
			raise ValueError(
				"`model.attention_mode=flex-block-attn` is not supported because "
				"reference and geometry tokens break the pretrained 3D sparse mask layout."
			)
		if not 0.0 <= self.condition_dropout_prob <= 1.0:
			raise ValueError(
				"`model.condition_dropout_prob` must be in [0, 1], "
				f"got {self.condition_dropout_prob}."
			)
		for name, value in (
			("drop_vae_tokens_prob", self.drop_vae_tokens_prob),
			("drop_vis_tokens_prob", self.drop_vis_tokens_prob),
			("drop_geo_tokens_prob", self.drop_geo_tokens_prob),
			("drop_txt_tokens_prob", self.drop_txt_tokens_prob),
		):
			if not 0.0 <= value <= 1.0:
				raise ValueError(
					f"`model.{name}` must be in [0, 1], "
					f"got {value}."
				)
		if self.cfg_guidance_scale < 1.0:
			raise ValueError(
				"`model.cfg_guidance_scale` must be at least 1.0, "
				f"got {self.cfg_guidance_scale}."
			)
		if not 0.0 <= self.cfg_guidance_rescale <= 1.0:
			raise ValueError(
				"`model.cfg_guidance_rescale` must be in [0, 1], "
				f"got {self.cfg_guidance_rescale}."
			)
		if self.lora_rank <= 0:
			raise ValueError(f"`model.lora_rank` must be positive, got {self.lora_rank}.")
		if self.geo_stream_init not in {"copy_image", "fresh"}:
			raise ValueError(
				"`model.geo_stream_init` must be either `copy_image` or `fresh`, "
				f"got {self.geo_stream_init!r}."
			)
		if self.num_train_timesteps <= 0:
			raise ValueError(
				"`model.num_train_timesteps` must be positive, "
				f"got {self.num_train_timesteps}."
			)
		if self.train_timestep_shift <= 0:
			raise ValueError(
				"`model.train_timestep_shift` must be positive, "
				f"got {self.train_timestep_shift}."
			)
		if self.eval_timestep_shift <= 0:
			raise ValueError(
				"`model.validation_timestep_shift` must be positive, "
				f"got {self.eval_timestep_shift}."
			)
		if self.inference_steps <= 0:
			raise ValueError(f"`model.inference_steps` must be positive, got {self.inference_steps}.")
		if self.snr_type not in {"uniform", "lognorm", "mix", "mode"}:
			raise ValueError(
				"`model.snr_type` must be one of `uniform`, `lognorm`, `mix`, or `mode`, "
				f"got {self.snr_type!r}."
			)


@dataclass
class DeepWorldQWConfig:
	"""Final composed config for the Qwen/Wan DeepWorld variant."""

	dataset: DatasetConfig = field(default_factory=DatasetConfig)
	optimizer: DeepWorldQWOptimizerConfig = field(default_factory=DeepWorldQWOptimizerConfig)
	training: TrainingConfig = field(default_factory=TrainingConfig)
	brain: DeepWorldQWBrainConfig = field(default_factory=DeepWorldQWBrainConfig)
	renderer: DeepWorldQWRendererConfig = field(default_factory=DeepWorldQWRendererConfig)


@dataclass
class DeepWorldHYConfig:
	"""Final composed config for the Hunyuan DeepWorld variant."""

	dataset: DatasetConfig = field(default_factory=DatasetConfig)
	optimizer: DeepWorldHYOptimizerConfig = field(default_factory=DeepWorldHYOptimizerConfig)
	training: TrainingConfig = field(default_factory=TrainingConfig)
	model: DeepWorldHYModelConfig = field(default_factory=DeepWorldHYModelConfig)


T = TypeVar("T")


def dataclass_from_dict(cls: Type[T], payload: Dict[str, Any]) -> T:
	"""Instantiate a dataclass directly from a plain mapping."""

	valid_fields = {field.name for field in fields(cls)}
	unknown_fields = sorted(set(payload) - valid_fields)
	if unknown_fields:
		raise ValueError(f"Unknown {cls.__name__} config field(s): {', '.join(unknown_fields)}")
	return cls(**payload)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
	"""Load a YAML file into a plain dictionary."""

	with Path(path).open("r", encoding="utf-8") as handle:
		return yaml.safe_load(handle) or {}


def load_config_from_sections(path: str | Path, config_cls: Type[T], section_types: dict[str, Type[Any]]) -> T:
	"""Load a composed config dataclass from known YAML sections."""

	payload = load_yaml_config(path)
	valid_sections = {field.name for field in fields(config_cls)}
	unknown_sections = sorted(set(payload) - valid_sections)
	if unknown_sections:
		raise ValueError(f"Unknown root config section(s): {', '.join(unknown_sections)}")
	section_values = {
		section: dataclass_from_dict(section_types[section], payload.get(section, {}))
		for section in valid_sections
	}
	return config_cls(**section_values)


def load_qw_config(path: str | Path) -> DeepWorldQWConfig:
	"""Load a DeepWorldQW YAML config."""

	return load_config_from_sections(path, DeepWorldQWConfig, {
		"dataset": DatasetConfig,
		"optimizer": DeepWorldQWOptimizerConfig,
		"training": TrainingConfig,
		"brain": DeepWorldQWBrainConfig,
		"renderer": DeepWorldQWRendererConfig,
	})


def load_hy_config(path: str | Path) -> DeepWorldHYConfig:
	"""Load a DeepWorldHY YAML config."""

	payload = load_yaml_config(path)
	if "flow_matching" in payload:
		model_payload = payload.setdefault("model", {})
		for key, value in payload.pop("flow_matching").items():
			if key in model_payload:
				raise ValueError(
					f"`flow_matching.{key}` duplicates `model.{key}`; keep the setting under `model`."
				)
			model_payload[key] = value

	optimizer_payload = payload.get("optimizer", {})
	# TODO: Would you stop doing these? I have already told you, remove legacy code, **do not** maintain compatability with previous code.
	for old_name, new_name in (
		("transformer_learning_rate", "adapter_learning_rate"),
		("geometry_learning_rate", "geo_stream_learning_rate"),
	):
		if old_name in optimizer_payload:
			if new_name in optimizer_payload:
				raise ValueError(
					f"`optimizer.{old_name}` duplicates `optimizer.{new_name}`; use `{new_name}`."
				)
			optimizer_payload[new_name] = optimizer_payload.pop(old_name)

	model_payload = payload.get("model", {})
	for old_name, new_name in (
		("use_reference_vae_tokens", "use_vae_tokens"),
		("use_siglip_tokens", "use_vis_tokens"),
		("use_geometry_tokens", "use_geo_tokens"),
	):
		if old_name in model_payload:
			if new_name in model_payload:
				raise ValueError(
					f"`model.{old_name}` duplicates `model.{new_name}`; use `{new_name}`."
				)
			model_payload[new_name] = model_payload.pop(old_name)

	valid_sections = {field.name for field in fields(DeepWorldHYConfig)}
	unknown_sections = sorted(set(payload) - valid_sections)
	if unknown_sections:
		raise ValueError(f"Unknown root config section(s): {', '.join(unknown_sections)}")
	return DeepWorldHYConfig(
		dataset=dataclass_from_dict(DatasetConfig, payload.get("dataset", {})),
		optimizer=dataclass_from_dict(DeepWorldHYOptimizerConfig, optimizer_payload),
		training=dataclass_from_dict(TrainingConfig, payload.get("training", {})),
		model=dataclass_from_dict(DeepWorldHYModelConfig, model_payload),
	)
