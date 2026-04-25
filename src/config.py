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
		dataset_source: Dataset backend, either `manifest` or `webdataset`.
		manifest_path: JSONL manifest describing the training set.
		webdataset_urls: Absolute local tar-shard paths or brace-expand patterns for WebDataset loading.
		num_samples: Optional deterministic per-epoch sample cap. For manifest datasets, `0` means all records. For WebDataset, this must be positive and divisible by distributed world size because it defines both the stream cap and reported length.
		num_reference_images: Maximum number of reference images to load per sample.
		video_num_frames: Target number of frames to sample from each GT video.
		video_frame_stride: Temporal stride used when sampling frames from the GT clip.
		video_height: Spatial height used for GT video training clips.
		video_width: Spatial width used for GT video training clips.
		vis_image_size: Square reference-image size used for Qwen's visual encoder.
		geo_image_size: Optional square reference-image size used for VGGT. If omitted, it is derived from `vis_image_size` by rounding down to a multiple of VGGT's patch size.
		max_text_length: Maximum prompt token length after tokenization.
		shuffle: Whether to use randomized temporal clips during training.
		num_workers: Number of dataloader workers.
		prefetch_factor: Number of batches loaded in advance by each worker.
		pin_memory: Whether to enable pinned-memory dataloader buffers.
		persistent_workers: Whether worker processes should stay alive across epochs.
	"""

	dataset_source: str = "manifest"
	manifest_path: str = ""
	webdataset_urls: List[str] | str = field(default_factory=list)
	num_samples: int = 0
	num_reference_images: int = 4
	video_num_frames: int = 81
	video_frame_stride: int = 1
	video_height: int = 480
	video_width: int = 832
	vis_image_size: int = 560
	geo_image_size: int | None = None
	max_text_length: int = 512
	shuffle: bool = True
	num_workers: int = 8
	prefetch_factor: int | None = None
	pin_memory: bool = True
	persistent_workers: bool = True


@dataclass
class OptimizerConfig:
	"""Optimizer and LR-schedule hyperparameters.

	Attributes:
		learning_rate: Base learning rate used by AdamW.
		weight_decay: Weight decay coefficient.
		betas: Adam beta coefficients.
		eps: Adam epsilon.
		max_grad_norm: Gradient clipping threshold applied after accumulation.
		warmup_steps: Number of warmup steps before cosine decay.
	"""

	learning_rate: float = 2e-5
	weight_decay: float = 1e-2
	betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
	eps: float = 1e-8
	max_grad_norm: float = 1.0
	warmup_steps: int = 500


@dataclass
class TrainingConfig:
	"""Top-level training loop configuration.

	Attributes:
		output_dir: Directory used for logs and checkpoints.
		seed: Global random seed.
		num_epochs: Number of training epochs.
		per_device_batch_size: Batch size on each process/GPU.
		gradient_accumulation_steps: Number of steps to accumulate before optimizer update.
		log_every: Logging interval in optimizer steps.
		save_every: Checkpoint interval in optimizer steps.
		mixed_precision: Accelerate precision mode such as `bf16`.
		use_fsdp: Whether multi-process training should use FSDP instead of DDP.
		gradient_checkpointing: Whether model submodules enable checkpointing when supported.
	"""

	output_dir: str = "outputs/world_model"
	seed: int = 42
	num_epochs: int = 10
	per_device_batch_size: int = 1
	gradient_accumulation_steps: int = 1
	log_every: int = 10
	save_every: int = 1000
	mixed_precision: str = "bf16"
	use_fsdp: bool = True
	gradient_checkpointing: bool = True


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
		train_scheduler_steps: Number of training diffusion timesteps. If omitted,
			inferred from the checkpoint scheduler config when available.
		inference_steps: Default number of denoising steps during sampling.
	"""

	checkpoint_path: str = "checkpoints/Wan2.1-T2V-1.3B-Diffusers"
	transformer_dtype: str | None = "bfloat16"
	vae_dtype: str | None = None
	vae_enable_slicing: bool = False
	vae_enable_tiling: bool = False
	train_scheduler_steps: int | None = None
	inference_steps: int = 50


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
