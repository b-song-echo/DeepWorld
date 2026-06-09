from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar

import yaml


def _resolve_lora_alpha(alpha: int | None, rank: int) -> int:
	"""Resolve a LoRA alpha value, defaulting to `2 * rank` when omitted.

	Args:
		alpha: Optional configured LoRA alpha value.
		rank: LoRA rank used by the same adapter.

	Returns:
		The explicit alpha value, or the conventional `2 * rank` default.
	"""

	return 2 * rank if alpha is None else int(alpha)


@dataclass
class DatasetConfig:
	"""Dataset and dataloader settings for world-model training.

	Attributes:
		manifest_path: JSONL manifest describing the generated training split.
		eval_manifest_path: JSONL manifest describing the generated validation split.
		data_root: Optional root used to resolve relative sample paths. If omitted, the manifest directory is used.
		num_samples: Optional deterministic per-epoch sample cap. `0` means all records.
		video_width: Spatial width used for GT video training clips and square reference preprocessing.
		video_height: Spatial height used for GT video training clips and square reference preprocessing.
		video_fps: Uniform FPS used to sample frames from each curated GT clip.
		video_duration: Duration in seconds represented by the curated clip.
		prompt_rich_prob: Probability of using the full synthesized prompt.
		prompt_medium_prob: Probability of using the medium distilled prompt.
		prompt_coarse_prob: Probability of using the coarse distilled prompt.
		vis_image_size: Square reference-image size used for Qwen's visual encoder.
		geo_image_size: Optional square reference-image size used for VGGT. If omitted, it is derived from `vis_image_size` by rounding down to a multiple of VGGT's patch size.
		max_text_length: Maximum prompt token length after tokenization.
		shuffle: Whether to use randomized temporal clips during training.
		num_workers: Number of dataloader workers.
		prefetch_factor: Number of batches loaded in advance by each worker.
		pin_memory: Whether to enable pinned-memory dataloader buffers.
		persistent_workers: Whether worker processes should stay alive across epochs.
	"""

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
	vis_image_size: int = 448
	geo_image_size: int | None = None
	max_text_length: int = 1024
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
			raise ValueError(f"`dataset.width` and `dataset.height` must be positive, got {(self.video_width, self.video_height)}.")
		if self.video_fps <= 0:
			raise ValueError(f"`dataset.video_fps` must be positive, got {self.video_fps}.")
		if self.video_duration <= 0:
			raise ValueError(f"`dataset.video_duration` must be positive, got {self.video_duration}.")
		if self.vis_image_size <= 0:
			raise ValueError(f"`dataset.vis_image_size` must be positive, got {self.vis_image_size}.")
		for name, value in (
			("prompt_rich_prob", self.prompt_rich_prob),
			("prompt_medium_prob", self.prompt_medium_prob),
			("prompt_coarse_prob", self.prompt_coarse_prob),
		):
			if value < 0:
				raise ValueError(f"`dataset.{name}` must be non-negative, got {value}.")
		if self.prompt_rich_prob + self.prompt_medium_prob + self.prompt_coarse_prob <= 0:
			raise ValueError("At least one prompt probability must be positive.")

	@property
	def video_num_frames(self) -> int:
		"""Return the Wan-aligned endpoint-inclusive frame count for the clip."""

		return int(round(self.video_fps * self.video_duration)) + 1


@dataclass
class OptimizerConfig:
	"""Optimizer and LR-schedule hyperparameters.

	Attributes:
		learning_rate: Base learning rate used by AdamW.
		brain_language_model_learning_rate: Learning rate for trainable Qwen language-model parameters. If omitted, defaults to `learning_rate`.
		brain_others_learning_rate: Learning rate for other trainable brain parameters. If omitted, defaults to `brain_language_model_learning_rate`.
		renderer_transformer_learning_rate: Learning rate for trainable Wan transformer parameters. If omitted, defaults to `learning_rate`.
		renderer_others_learning_rate: Learning rate for other trainable renderer parameters. If omitted, defaults to `renderer_transformer_learning_rate`.
		lr_schedule: Learning-rate schedule after warmup, either `cosine` or `constant`.
		weight_decay: Weight decay coefficient.
		betas: Adam beta coefficients.
		eps: Adam epsilon.
		max_grad_norm: Gradient clipping threshold applied after accumulation.
		warmup_steps: Number of warmup steps before the requested post-warmup schedule.
	"""

	learning_rate: float = 2e-5
	brain_language_model_learning_rate: float | None = None
	brain_others_learning_rate: float | None = None
	renderer_transformer_learning_rate: float | None = None
	renderer_others_learning_rate: float | None = None
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
class TrainingConfig:
	"""Top-level training loop configuration.

	Attributes:
		output_dir: Directory used for logs and checkpoints.
		seed: Global random seed.
		max_total_epochs: Optional maximum number of epochs to train.
		max_total_steps: Optional maximum number of optimizer steps to train.
		per_device_batch_size: Batch size on each process/GPU.
		renderer_batch_multiplier: Number of independent diffusion timesteps to train per encoded video.
		gradient_accumulation_steps: Number of steps to accumulate before optimizer update.
		log_every: Logging interval in optimizer steps.
		save_every: Checkpoint interval in optimizer steps.
		eval_num_samples: Number of training-set samples to generate before each checkpoint save. `0` disables evaluation generation.
		mixed_precision: Accelerate precision mode such as `bf16`.
		use_fsdp: Whether multi-process training should use FSDP instead of DDP.
		gradient_checkpointing: Whether model submodules enable checkpointing when supported.
		pretrained_model_path: Optional path to a saved `model.pt` state dict to load before training.
	"""

	output_dir: str = "outputs/world_model"
	seed: int = 42
	max_total_epochs: int | None = 10
	max_total_steps: int | None = None
	per_device_batch_size: int = 1
	renderer_batch_multiplier: int = 1
	gradient_accumulation_steps: int = 1
	log_every: int = 10
	save_every: int = 1000
	eval_num_samples: int = 0
	mixed_precision: str = "bf16"
	use_fsdp: bool = True
	gradient_checkpointing: bool = True
	pretrained_model_path: str | None = None

	def __post_init__(self) -> None:
		"""Validate training-loop settings and bounded stopping criteria."""

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
		if self.per_device_batch_size < 1:
			raise ValueError(
				f"`training.per_device_batch_size` must be at least 1, got {self.per_device_batch_size}."
			)
		if self.renderer_batch_multiplier < 1:
			raise ValueError(
				f"`training.renderer_batch_multiplier` must be at least 1, got {self.renderer_batch_multiplier}."
			)
		if self.gradient_accumulation_steps < 1:
			raise ValueError(
				"`training.gradient_accumulation_steps` must be at least 1, "
				f"got {self.gradient_accumulation_steps}."
			)
		if self.eval_num_samples < 0:
			raise ValueError(f"`training.eval_num_samples` must be non-negative, got {self.eval_num_samples}.")


@dataclass
class QwenBrainConfig:
	"""Configuration for the Qwen brain branch.

	Attributes:
		checkpoint_path: Local path to the Qwen3-VL checkpoint directory.
		transformer_dtype: Requested Qwen language/vision load dtype. If omitted, the checkpoint default is used.
		vggt_checkpoint_path: Local path to the VGGT checkpoint directory.
		vggt_dtype: Optional dtype applied to VGGT after loading. If omitted, the checkpoint default is used.
		lora_rank: LoRA rank used for language-model attention projections.
		lora_alpha: Optional LoRA scaling factor. If omitted, defaults to `2 * lora_rank`.
		lora_dropout: LoRA dropout probability.
		lora_target_modules: Linear module names that should receive LoRA adapters.
		routed_ffn_mode: Training mode for the three modality FFNs, either `full` or `lora`.
		routed_ffn_lora_rank: LoRA rank used when `routed_ffn_mode=lora`.
		routed_ffn_lora_alpha: Optional LoRA scaling factor for routed FFNs. If omitted, defaults to `2 * routed_ffn_lora_rank`.
		routed_ffn_lora_dropout: LoRA dropout probability for routed FFNs.
	"""

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
		self.lora_alpha = _resolve_lora_alpha(self.lora_alpha, self.lora_rank)
		self.routed_ffn_lora_alpha = _resolve_lora_alpha(self.routed_ffn_lora_alpha, self.routed_ffn_lora_rank)


@dataclass
class WanRendererConfig:
	"""Configuration for the Wan renderer branch.

	Attributes:
		checkpoint_path: Local path to the Wan diffusers checkpoint directory.
		transformer_dtype: Requested Wan transformer load dtype. If omitted, the checkpoint default is used.
		vae_dtype: Optional Wan VAE load dtype. If omitted, the checkpoint default is used.
		vae_enable_slicing: Whether to enable diffusers VAE slicing for lower memory use.
		vae_enable_tiling: Whether to enable diffusers VAE tiling for lower memory use.
		vae_sample_posterior: Whether the frozen VAE samples from the latent posterior during training instead of using its mode.
		condition_injection_mode: How Qwen states condition Wan, either `input_addition` or `cross_attention`.
		condition_proj_init: Initialization strategy for Qwen-to-Wan conditioning projection, either `zero`, `normal`, or `default`.
		condition_proj_init_std: Optional standard deviation used when `condition_proj_init=normal`. If omitted, defaults to `1e-3`.
		condition_dropout_prob: Per-renderer-sample probability of replacing all conditioning tokens with the learned null condition during training.
		train_scheduler_steps: Number of training diffusion timesteps. If omitted,
			inferred from the checkpoint scheduler config when available.
		inference_steps: Default number of denoising steps during sampling.
	"""

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
				"`wan_renderer.condition_injection_mode` must be `input_addition` or `cross_attention`, "
				f"got {self.condition_injection_mode!r}."
			)
		self.condition_proj_init = self.condition_proj_init.lower()
		if self.condition_proj_init not in {"zero", "normal", "default"}:
			raise ValueError(
				"`wan_renderer.condition_proj_init` must be `zero`, `normal`, or `default`, "
				f"got {self.condition_proj_init!r}."
			)
		if self.condition_proj_init == "normal" and self.condition_proj_init_std is None:
			self.condition_proj_init_std = 1e-3
		if self.condition_proj_init_std is not None and self.condition_proj_init_std <= 0:
			raise ValueError(
				"`wan_renderer.condition_proj_init_std` must be positive, "
				f"got {self.condition_proj_init_std}."
			)
		if not 0.0 <= self.condition_dropout_prob <= 1.0:
			raise ValueError(
				"`wan_renderer.condition_dropout_prob` must be in [0, 1], "
				f"got {self.condition_dropout_prob}."
			)


@dataclass
class WorldModelConfig:
	"""Root configuration object used by the prototype."""

	dataset: DatasetConfig = field(default_factory=DatasetConfig)
	optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
	training: TrainingConfig = field(default_factory=TrainingConfig)
	qwen_brain: QwenBrainConfig = field(default_factory=QwenBrainConfig)
	wan_renderer: WanRendererConfig = field(default_factory=WanRendererConfig)


T = TypeVar("T")


def _dataclass_from_dict(cls: Type[T], payload: Dict[str, Any]) -> T:
	"""Instantiate a dataclass directly from a plain mapping.

	Args:
		cls: Dataclass type to instantiate.
		payload: Field-value mapping loaded from YAML.

	Returns:
		An instance of `cls`.
	"""

	valid_fields = {field.name for field in fields(cls)}
	unknown_fields = sorted(set(payload) - valid_fields)
	if unknown_fields:
		raise ValueError(f"Unknown {cls.__name__} config field(s): {', '.join(unknown_fields)}")
	return cls(**payload)


def load_config(path: str | Path) -> WorldModelConfig:
	"""Load the YAML config file used by training and model construction.

	Args:
		path: Path to a YAML config file.

	Returns:
		A fully constructed `WorldModelConfig` with nested dataclass sections.
	"""

	with Path(path).open("r", encoding="utf-8") as handle:
		payload = yaml.safe_load(handle) or {}
	valid_sections = {field.name for field in fields(WorldModelConfig)}
	unknown_sections = sorted(set(payload) - valid_sections)
	if unknown_sections:
		raise ValueError(f"Unknown root config section(s): {', '.join(unknown_sections)}")
	return WorldModelConfig(
		dataset=_dataclass_from_dict(DatasetConfig, payload.get("dataset", {})),
		optimizer=_dataclass_from_dict(OptimizerConfig, payload.get("optimizer", {})),
		training=_dataclass_from_dict(TrainingConfig, payload.get("training", {})),
		qwen_brain=_dataclass_from_dict(QwenBrainConfig, payload.get("qwen_brain", {})),
		wan_renderer=_dataclass_from_dict(WanRendererConfig, payload.get("wan_renderer", {})),
	)
