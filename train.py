import os
import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import TextIO

import torch
import yaml
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset, Subset
from tqdm.auto import tqdm
from transformers import AutoProcessor, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from src.config import load_config, WorldModelConfig
from src.data import WorldModelBatchCollator, build_dataset
from src.models import DeepWorld
from src.utils.compat import get_world_size
from src.utils.video import save_image_tensor, save_video_tensor


VIDEO_DURATION_SECONDS = 10.0


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for training.

	Returns:
		Namespace containing the required config path.
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--config", type=str, required=True,
		help="Path to a YAML config file."
	)
	return parser.parse_args()


def align_fsdp_managed_dtypes(model: DeepWorld) -> None:
	"""Align non-ignored module dtypes for FSDP flattening.

	FSDP requires each flattened parameter group to use one dtype. Frozen encoders
	that intentionally keep their own dtype are ignored by FSDP; all remaining
	trainable and wrapped modules are cast to the Qwen language-model dtype.

	Args:
		model: Unprepared world-model instance.
	"""

	target_dtype = next(model.brain.language_model.parameters()).dtype
	managed_modules = (
		model.brain.language_model,
		model.brain.geo_bridge,
		model.brain.segment_tokens,
		model.renderer.transformer,
		model.renderer.condition_proj,
	)
	managed_parameters = (
		model.brain.gen_slot_token,
		model.renderer.null_cond,
	)
	managed_dtypes = {
		parameter.dtype
		for module in managed_modules
		for parameter in module.parameters(recurse=True)
	}
	managed_dtypes.update(parameter.dtype for parameter in managed_parameters)
	if managed_dtypes <= {target_dtype}:
		return

	for module in managed_modules:
		module.to(dtype=target_dtype)
	for parameter in managed_parameters:
		with torch.no_grad():
			parameter.data = parameter.data.to(dtype=target_dtype)


def create_accelerator(
	model: DeepWorld,
	output_dir: str,
	grad_acc_steps: int,
	mixed_precision: str,
	use_fsdp: bool,
) -> Accelerator:
	"""Create the Accelerate runtime and optional in-code FSDP plugin.

	Args:
		model: Unprepared world-model instance.
		output_dir: Directory used for tracker output.
		grad_acc_steps: Gradient accumulation steps.
		mixed_precision: Mixed precision mode such as `bf16`.
		use_fsdp: Whether multi-process training should use FSDP instead of DDP.

	Returns:
		A configured `Accelerator`.
	"""

	world_size = get_world_size()
	fsdp_enabled = use_fsdp and world_size > 1
	if fsdp_enabled:
		align_fsdp_managed_dtypes(model)

	ignored_modules = [
		model.renderer.vae,
		model.brain.geometry_encoder,
		model.brain.vision_encoder,
	]

	fsdp_plugin = FullyShardedDataParallelPlugin(
		fsdp_version=1,
		sharding_strategy="FULL_SHARD",
		auto_wrap_policy="TRANSFORMER_BASED_WRAP",
		transformer_cls_names_to_wrap=["MoFfnQwenDecoderLayer"],
		state_dict_type="FULL_STATE_DICT",
		backward_prefetch="BACKWARD_PRE",
		forward_prefetch=False,
		activation_checkpointing=False,
		ignored_modules=ignored_modules,
		use_orig_params=True,
		limit_all_gathers=True,
		cpu_ram_efficient_loading=True,
		sync_module_states=True,
		cpu_offload=False
	) if fsdp_enabled else None

	return Accelerator(
		gradient_accumulation_steps=grad_acc_steps,
		mixed_precision=mixed_precision,
		project_dir=output_dir,
		fsdp_plugin=fsdp_plugin,
	)


def _count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> tuple[int, int]:
	"""Return parameter count and byte size for one module tree."""

	num_parameters = 0
	num_bytes = 0
	for parameter in module.parameters():
		if trainable_only and not parameter.requires_grad:
			continue
		num_parameters += parameter.numel()
		num_bytes += parameter.numel() * parameter.element_size()
	return num_parameters, num_bytes


def _format_count(value: int) -> str:
	"""Format large integer counts for startup logging."""

	for unit in ["", "K", "M", "B", "T"]:
		if abs(value) < 1000 or unit == "T":
			return f"{value:.2f}{unit}" if unit else str(value)
		value /= 1000
	return str(value)


def _format_bytes(value: int) -> str:
	"""Format byte counts for startup logging."""

	for unit in ["B", "KB", "MB", "GB", "TB"]:
		if abs(value) < 1024 or unit == "TB":
			return f"{value:.2f}{unit}" if unit != "B" else f"{value}B"
		value /= 1024
	return f"{value:.2f}TB"


def log_parameter_summary(accelerator: Accelerator, model: DeepWorld) -> None:
	"""Print a compact parameter and static-weight summary for memory debugging."""

	total_parameters, total_bytes = _count_parameters(model)
	trainable_parameters, trainable_bytes = _count_parameters(model, trainable_only=True)
	accelerator.print(
		"Model parameters: "
		f"total={_format_count(total_parameters)} ({_format_bytes(total_bytes)}), "
		f"trainable={_format_count(trainable_parameters)} ({_format_bytes(trainable_bytes)})."
	)
	for name, module in (
		("brain.language_model", model.brain.language_model),
		("brain.vision_encoder", model.brain.vision_encoder),
		("brain.geometry_encoder", model.brain.geometry_encoder),
		("renderer.transformer", model.renderer.transformer),
		("renderer.vae", model.renderer.vae),
	):
		module_parameters, module_bytes = _count_parameters(module)
		module_trainable, _ = _count_parameters(module, trainable_only=True)
		accelerator.print(
			f"  {name}: {_format_count(module_parameters)} params, {_format_bytes(module_bytes)}, "
			f"trainable={_format_count(module_trainable)}"
		)


def log_runtime_setup(accelerator: Accelerator, model: torch.nn.Module) -> None:
	"""Log the prepared distributed runtime and whether FSDP wrapping is active."""

	distributed_type = getattr(accelerator.distributed_type, "name", str(accelerator.distributed_type))
	accelerator.print(
		f"Accelerate runtime: distributed_type={distributed_type}, world_size={accelerator.num_processes}, "
		f"device={accelerator.device}."
	)
	if distributed_type != "FSDP":
		return

	is_fsdp_wrapped = False
	try:
		from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
		is_fsdp_wrapped = isinstance(model, FSDP)
	except Exception:
		pass
	accelerator.print(
		f"FSDP active={is_fsdp_wrapped}. Ignored branches: "
		"`renderer.vae`, `brain.geometry_encoder`, `brain.vision_encoder`."
	)


def save_config_snapshot(config: WorldModelConfig, output_dir: Path, accelerator: Accelerator) -> None:
	"""Write the resolved training config beside logs and checkpoints.

	Args:
		config: Fully resolved root configuration object.
		output_dir: Directory where the config snapshot should be written.
		accelerator: Active Accelerate runtime used to restrict writing to rank 0.
	"""

	if accelerator.is_main_process:
		config_path = output_dir / "configs.yaml"
		with config_path.open("w", encoding="utf-8") as handle:
			yaml.safe_dump(asdict(config), handle, sort_keys=False)
	accelerator.wait_for_everyone()


def open_jsonl_log(output_dir: Path, accelerator: Accelerator) -> TextIO | None:
	"""Open the rank-0 JSONL metrics log.

	Args:
		output_dir: Training output directory.
		accelerator: Active Accelerate runtime used to restrict logging to rank 0.

	Returns:
		An open text handle for rank 0, otherwise `None`.
	"""

	if not accelerator.is_main_process:
		return None
	return (output_dir / "logs.jsonl").open("a", encoding="utf-8", buffering=1)


def write_jsonl_log(handle: TextIO | None, payload: dict) -> None:
	"""Append one JSONL entry and force it to disk.

	Args:
		handle: Optional rank-0 log handle.
		payload: JSON-serializable log entry.
	"""

	if handle is None:
		return
	handle.write(json.dumps(payload, sort_keys=True) + "\n")
	handle.flush()
	os.fsync(handle.fileno())


def build_lr_scheduler(optimizer: AdamW, config: WorldModelConfig, total_training_steps: int):
	"""Create the configured warmup learning-rate scheduler.

	Args:
		optimizer: Optimizer whose learning rate should be scheduled.
		config: Root training configuration.
		total_training_steps: Total number of optimizer steps in this run.

	Returns:
		A Transformers scheduler with warmup and the requested post-warmup behavior.
	"""

	if config.optimizer.lr_schedule == "constant":
		return get_constant_schedule_with_warmup(
			optimizer,
			num_warmup_steps=config.optimizer.warmup_steps,
		)
	if config.optimizer.lr_schedule == "cosine":
		return get_cosine_schedule_with_warmup(
			optimizer,
			num_warmup_steps=config.optimizer.warmup_steps,
			num_training_steps=total_training_steps,
		)
	raise ValueError(f"Unsupported optimizer.lr_schedule: {config.optimizer.lr_schedule!r}.")


# TODO: Don't put this in a function... It is only a single line of code, why do you have to put everything in a function...

def _trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
	"""Collect trainable parameters from one module tree.

	Args:
		module: Module whose parameters should be inspected.

	Returns:
		A list of parameters requiring gradients.
	"""

	return [parameter for parameter in module.parameters() if parameter.requires_grad]


# TODO: make this a helper nested function inside build_optimizer, no need to document
def _other_trainable_parameters(
	module: torch.nn.Module,
	excluded_parameter_ids: set[int],
) -> list[torch.nn.Parameter]:
	"""Collect trainable parameters outside an already-owned parameter set.

	Args:
		module: Module whose parameters should be inspected.
		excluded_parameter_ids: Parameter identities that already belong to a more specific group.

	Returns:
		A list of trainable parameters not present in `excluded_parameter_ids`.
	"""

	return [
		parameter
		for parameter in module.parameters()
		if parameter.requires_grad and id(parameter) not in excluded_parameter_ids
	]


# TODO: make this a helper nested function inside build_optimizer, no need to document

def _add_optimizer_group(
	parameter_groups: list[dict],
	group_name: str,
	parameters: list[torch.nn.Parameter],
	learning_rate: float,
) -> None:
	"""Append one non-empty AdamW parameter group.

	Args:
		parameter_groups: Mutable optimizer group list.
		group_name: Human-readable group name stored in the optimizer state.
		parameters: Parameters assigned to this group.
		learning_rate: Learning rate for this group.
	"""

	if len(parameters) == 0:
		return
	parameter_groups.append({
		"name": group_name,
		"params": parameters,
		"lr": learning_rate,
	})


def _validate_optimizer_groups(model: DeepWorld, grouped_parameters: list[torch.nn.Parameter]) -> None:
	"""Ensure optimizer grouping covers each trainable parameter exactly once.

	Args:
		model: Unprepared model whose trainable parameters should be optimized.
		grouped_parameters: Flattened parameter list assigned to optimizer groups.

	Raises:
		RuntimeError: If any trainable parameter is missing or assigned more than once.
	"""

	trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
	grouped_ids = [id(parameter) for parameter in grouped_parameters]
	grouped_id_set = set(grouped_ids)
	if len(grouped_ids) != len(grouped_id_set):
		raise RuntimeError("Optimizer parameter groups contain duplicate trainable parameters.")
	if trainable_ids != grouped_id_set:
		missing_count = len(trainable_ids - grouped_id_set)
		extra_count = len(grouped_id_set - trainable_ids)
		raise RuntimeError(
			"Optimizer parameter groups do not match the model's trainable parameters: "
			f"missing={missing_count}, extra={extra_count}."
		)


def build_optimizer(model: DeepWorld, config: WorldModelConfig) -> AdamW:
	"""Build AdamW with branch-specific learning-rate groups.

	The four requested groups are:
	- trainable parameters in `model.brain.language_model`,
	- other trainable parameters in `model.brain`,
	- trainable parameters in `model.renderer.transformer`,
	- other trainable parameters in `model.renderer`.

	Args:
		model: Unprepared model whose trainable parameters should be optimized.
		config: Root config containing optimizer hyperparameters and group LRs.

	Returns:
		An AdamW optimizer with one parameter group per non-empty branch group.
	"""

	qwen_language_parameters = _trainable_parameters(model.brain.language_model)
	qwen_language_ids = {id(parameter) for parameter in qwen_language_parameters}
	qwen_other_parameters = _other_trainable_parameters(
		model.brain,
		qwen_language_ids,
	)

	wan_transformer_parameters = _trainable_parameters(model.renderer.transformer)
	wan_transformer_ids = {id(parameter) for parameter in wan_transformer_parameters}
	wan_other_parameters = _other_trainable_parameters(
		model.renderer,
		wan_transformer_ids,
	)

	grouped_parameters = (
		qwen_language_parameters
		+ qwen_other_parameters
		+ wan_transformer_parameters
		+ wan_other_parameters
	)
	_validate_optimizer_groups(model, grouped_parameters)

	parameter_groups: list[dict] = []
	_add_optimizer_group(
		parameter_groups,
		"qwen_language_model",
		qwen_language_parameters,
		config.optimizer.qwen_language_learning_rate,
	)
	_add_optimizer_group(
		parameter_groups,
		"qwen_other",
		qwen_other_parameters,
		config.optimizer.qwen_other_learning_rate,
	)
	_add_optimizer_group(
		parameter_groups,
		"wan_transformer",
		wan_transformer_parameters,
		config.optimizer.wan_transformer_learning_rate,
	)
	_add_optimizer_group(
		parameter_groups,
		"wan_other",
		wan_other_parameters,
		config.optimizer.wan_other_learning_rate,
	)
	if len(parameter_groups) == 0:
		raise RuntimeError("No trainable parameters were found for optimizer construction.")

	return AdamW(
		parameter_groups,
		weight_decay=config.optimizer.weight_decay,
		betas=tuple(config.optimizer.betas),
		eps=config.optimizer.eps,
	)


def create_world_dataloader(
	dataset,
	dataset_config,
	batch_size: int,
	collator: WorldModelBatchCollator,
	shuffle: bool,
) -> DataLoader:
	"""Build a DataLoader while respecting PyTorch worker argument constraints.

	Args:
		dataset: Dataset object to iterate.
		dataset_config: Dataset configuration that controls worker and memory settings.
		batch_size: Number of samples per dataloader batch.
		collator: Collation function for raw world-model samples.
		shuffle: Whether the map-style dataset should be shuffled.

	Returns:
		A configured PyTorch dataloader.
	"""

	loader_kwargs = {
		"dataset": dataset,
		"batch_size": batch_size,
		"shuffle": shuffle,
		"num_workers": dataset_config.num_workers,
		"pin_memory": dataset_config.pin_memory and torch.cuda.is_available(),
		"persistent_workers": dataset_config.persistent_workers and dataset_config.num_workers > 0,
		"collate_fn": collator,
	}
	if dataset_config.num_workers > 0 and dataset_config.prefetch_factor is not None:
		loader_kwargs["prefetch_factor"] = dataset_config.prefetch_factor
	return DataLoader(**loader_kwargs)


def build_eval_dataloader(
	config: WorldModelConfig,
	collator: WorldModelBatchCollator,
	accelerator: Accelerator,
) -> DataLoader | None:
	"""Build a deterministic per-rank evaluation loader over a training-data prefix.

	Args:
		config: Root training configuration.
		collator: Batch collator shared with training.
		accelerator: Active Accelerate runtime used for rank partitioning.

	Returns:
		A rank-local eval dataloader, or `None` when eval generation is disabled.
	"""

	eval_num_samples = config.training.eval_num_samples
	if eval_num_samples == 0:
		return None
	if eval_num_samples % accelerator.num_processes != 0:
		raise ValueError(
			"`training.eval_num_samples` must be divisible by the distributed world size; "
			f"got eval_num_samples={eval_num_samples}, world_size={accelerator.num_processes}."
		)
	if eval_num_samples < accelerator.num_processes:
		raise ValueError(
			"`training.eval_num_samples` must be either 0 or at least the distributed world size; "
			f"got eval_num_samples={eval_num_samples}, world_size={accelerator.num_processes}."
		)
	if config.dataset.num_samples > 0 and eval_num_samples > config.dataset.num_samples:
		raise ValueError(
			"`training.eval_num_samples` must not exceed `dataset.num_samples` when the training set is capped; "
			f"got eval_num_samples={eval_num_samples}, dataset.num_samples={config.dataset.num_samples}."
		)

	eval_dataset_config = replace(config.dataset, num_samples=eval_num_samples, shuffle=False)
	eval_dataset = build_dataset(eval_dataset_config)
	if isinstance(eval_dataset, IterableDataset):
		eval_source = eval_dataset
	else:
		if len(eval_dataset) < eval_num_samples:
			raise ValueError(
				"`training.eval_num_samples` exceeds the available evaluation dataset size; "
				f"got eval_num_samples={eval_num_samples}, available={len(eval_dataset)}."
			)
		rank_indices = list(range(accelerator.process_index, eval_num_samples, accelerator.num_processes))
		eval_source = Subset(eval_dataset, rank_indices)

	return create_world_dataloader(
		dataset=eval_source,
		dataset_config=eval_dataset_config,
		batch_size=1,
		collator=collator,
		shuffle=False,
	)


def create_checkpoint_dir(accelerator: Accelerator, step: int, output_dir: Path) -> Path:
	"""Create and synchronize one checkpoint directory before all ranks use it.

	Args:
		accelerator: Active Accelerate runtime.
		step: Global optimizer step used in the checkpoint name.
		output_dir: Root directory where checkpoints should be written.

	Returns:
		The synchronized checkpoint directory path.
	"""

	checkpoint_dir = output_dir / f"checkpoint-{step}"
	if accelerator.is_main_process:
		checkpoint_dir.mkdir(parents=True, exist_ok=True)
	accelerator.wait_for_everyone()
	return checkpoint_dir


def make_eval_generator(device: torch.device, seed: int) -> torch.Generator:
	"""Create a deterministic random generator on the active model device.

	Args:
		device: Device where random latents will be sampled.
		seed: Deterministic seed for the sample.

	Returns:
		A seeded torch random generator.
	"""

	generator = torch.Generator(device=device)
	generator.manual_seed(seed)
	return generator


def save_eval_sample_bundle(
	batch: dict,
	generated_video: torch.Tensor,
	batch_index: int,
	global_eval_index: int,
	checkpoint_dir: Path,
) -> None:
	"""Save one generated sample with its text and training inputs.

	Args:
		batch: Collated evaluation batch containing prompts, references, and GT video.
		generated_video: Generated video tensor with shape `(3, T, H, W)`.
		batch_index: Index of the sample inside the current dataloader batch.
		global_eval_index: Stable global eval-sample index across ranks.
		checkpoint_dir: Checkpoint directory where sample folders should be written.
	"""

	sample_dir = checkpoint_dir / f"sample_{global_eval_index:04d}"
	sample_dir.mkdir(parents=True, exist_ok=True)
	save_video_tensor(generated_video, sample_dir / "generated.mp4", duration_seconds=VIDEO_DURATION_SECONDS)
	save_video_tensor(batch["videos"][batch_index], sample_dir / "ground_truth.mp4", duration_seconds=VIDEO_DURATION_SECONDS)

	prompt = batch["prompts"][batch_index]
	(sample_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

	reference_dir = sample_dir / "references"
	reference_mask = batch["reference_mask"][batch_index]
	reference_images = batch["geo_images"][batch_index]
	for ref_index, is_valid in enumerate(reference_mask.tolist()):
		if not is_valid:
			continue
		save_image_tensor(reference_images[ref_index], reference_dir / f"reference_{ref_index:02d}.png")


def generate_checkpoint_samples(
	accelerator: Accelerator,
	model: torch.nn.Module,
	eval_dataloader: DataLoader | None,
	config: WorldModelConfig,
	step: int,
	checkpoint_dir: Path,
) -> None:
	"""Generate and save rank-local evaluation videos before checkpoint serialization.

	Args:
		accelerator: Active Accelerate runtime.
		model: Prepared model used for generation.
		eval_dataloader: Rank-local dataloader over the eval subset, or `None`.
		config: Root training configuration.
		step: Global optimizer step used for deterministic sample seeds.
		checkpoint_dir: Directory where generated videos should be saved.
	"""

	if eval_dataloader is None:
		return

	per_process_samples = config.training.eval_num_samples // accelerator.num_processes
	was_training = model.training
	model.eval()
	num_saved = 0
	try:
		for batch in eval_dataloader:
			if num_saved >= per_process_samples:
				break

			global_eval_index = num_saved * accelerator.num_processes + accelerator.process_index
			generator = make_eval_generator(
				accelerator.device,
				config.training.seed + step * 1000003 + global_eval_index,
			)
			with torch.no_grad():
				with accelerator.autocast():
					outputs = model(batch, generate_samples=True, generator=generator)

			videos = outputs["videos"].detach().cpu()
			for batch_index in range(videos.size(0)):
				if num_saved >= per_process_samples:
					break
				global_eval_index = num_saved * accelerator.num_processes + accelerator.process_index
				save_eval_sample_bundle(
					batch=batch,
					generated_video=videos[batch_index],
					batch_index=batch_index,
					global_eval_index=global_eval_index,
					checkpoint_dir=checkpoint_dir,
				)
				num_saved += 1

		if num_saved != per_process_samples:
			raise RuntimeError(
				"Evaluation dataloader ended before this rank generated its requested samples; "
				f"rank={accelerator.process_index}, saved={num_saved}, expected={per_process_samples}."
			)
	finally:
		if was_training:
			model.train()

	accelerator.wait_for_everyone()
	if accelerator.is_main_process:
		accelerator.print(
			f"Saved {config.training.eval_num_samples} evaluation videos to {checkpoint_dir}."
		)


def save_checkpoint(accelerator: Accelerator, model: torch.nn.Module, checkpoint_dir: Path) -> None:
	"""Save one model checkpoint on the main process.

	Args:
		accelerator: Active Accelerate runtime.
		model: Prepared model instance to serialize.
		checkpoint_dir: Directory where `model.pt` should be written.
	"""

	# NOTE: accelerator.get_state_dict must be called on every processes, or the program will hang because the main process is indefinitely waiting for others.
	accelerator.wait_for_everyone()
	state_dict = accelerator.get_state_dict(model)
	if accelerator.is_main_process:
		checkpoint_dir.mkdir(parents=True, exist_ok=True)
		torch.save(state_dict, checkpoint_dir / "model.pt")
	accelerator.wait_for_everyone()


def save_checkpoint_with_evaluation(
	accelerator: Accelerator,
	model: torch.nn.Module,
	eval_dataloader: DataLoader | None,
	config: WorldModelConfig,
	step: int,
	output_dir: Path,
) -> None:
	"""Generate evaluation videos and then save model weights in the same directory.

	Args:
		accelerator: Active Accelerate runtime.
		model: Prepared model instance to serialize.
		eval_dataloader: Rank-local eval dataloader, or `None`.
		config: Root training configuration.
		step: Global optimizer step used in the checkpoint name.
		output_dir: Root directory where checkpoints should be written.
	"""

	checkpoint_dir = create_checkpoint_dir(accelerator, step, output_dir)
	generate_checkpoint_samples(accelerator, model, eval_dataloader, config, step, checkpoint_dir)
	save_checkpoint(accelerator, model, checkpoint_dir)


def set_preliminaries() -> None:
	"""Apply process-wide runtime defaults before model construction."""

	os.environ.setdefault("HF_HUB_OFFLINE", "1")
	os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
	os.environ.setdefault("DIFFUSERS_OFFLINE", "1")

	import warnings
	warnings.filterwarnings("ignore")

	from diffusers.utils import logging
	logging.set_verbosity_error()

	torch.multiprocessing.set_sharing_strategy('file_system')
	if torch.cuda.is_available():
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = False


def main() -> None:
	"""Run the distributed training loop.

	The function performs config loading, dataloader construction, model setup,
	optimizer/scheduler creation, Accelerate preparation, logging, periodic
	evaluation generation, and checkpoint saving.
	"""

	set_preliminaries()
	args = parse_args()
	config = load_config(args.config)
	set_seed(config.training.seed)
	output_dir = Path(config.training.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	model = DeepWorld(config)
	accelerator = create_accelerator(
		model=model,
		output_dir=config.training.output_dir,
		grad_acc_steps=config.training.gradient_accumulation_steps,
		mixed_precision=config.training.mixed_precision,
		use_fsdp=config.training.use_fsdp,
	)
	save_config_snapshot(config, output_dir, accelerator)
	metrics_log = open_jsonl_log(output_dir, accelerator)
	write_jsonl_log(metrics_log, {"event": "start"})
	log_parameter_summary(accelerator, model)

	model.to(accelerator.device) # NOTE: do not remove it at the moment
	optimizer = build_optimizer(model, config)

	dataset = build_dataset(config.dataset)
	processor = AutoProcessor.from_pretrained(config.qwen_brain.checkpoint_path, local_files_only=True)
	collator = WorldModelBatchCollator(
		dataset_config=config.dataset,
		tokenizer=processor.tokenizer,
		image_processor=processor.image_processor,
		geo_patch_size=model.brain.geo_patch_size,
	)
	if collator.geo_image_size != config.dataset.vis_image_size:
		accelerator.print(
			f"Adjusted VGGT geometry image size from {config.dataset.vis_image_size} to {collator.geo_image_size} "
			f"to match patch size {collator.geo_patch_size}."
		)

	dataloader = create_world_dataloader(
		dataset=dataset,
		batch_size=config.training.per_device_batch_size,
		shuffle=(not isinstance(dataset, IterableDataset)) and config.dataset.shuffle,
		dataset_config=config.dataset,
		collator=collator,
	)
	eval_dataloader = build_eval_dataloader(config, collator, accelerator)

	if isinstance(dataset, IterableDataset):
		model, optimizer = accelerator.prepare(model, optimizer)
	else:
		model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
	log_runtime_setup(accelerator, model)
	grad_clip_parameters = tuple(parameter for group in optimizer.param_groups for parameter in group["params"])

	steps_per_epoch = math.ceil(len(dataloader) / config.training.gradient_accumulation_steps)
	total_training_steps = steps_per_epoch * config.training.num_epochs
	lr_scheduler = build_lr_scheduler(optimizer, config, total_training_steps)
	progress_bar = tqdm(
		total=total_training_steps,
		desc="Training",
		disable=not accelerator.is_local_main_process,
		dynamic_ncols=True,
	)
	progress_bar.set_postfix(step=0, epoch=0, loss="n/a")

	global_step = 0
	accumulated_losses: list[torch.Tensor] = []
	for epoch in range(config.training.num_epochs):
		model.train()
		for batch in dataloader:
			with accelerator.accumulate(model):
				with accelerator.autocast():
					loss = model(batch)["loss"]
				accumulated_losses.append(loss.detach().float())
				accelerator.backward(loss)
				if accelerator.sync_gradients:
					accelerator.clip_grad_norm_(grad_clip_parameters, config.optimizer.max_grad_norm)
				optimizer.step()
				optimizer.zero_grad(set_to_none=True)

				if accelerator.sync_gradients:
					lr_scheduler.step()
					global_step += 1
					macro_loss = torch.stack(accumulated_losses).mean()
					accumulated_losses.clear()
					loss_value = accelerator.gather(macro_loss).mean().item()
					
					progress_bar.update(1)
					progress_bar.set_postfix(step=global_step, epoch=epoch + 1, loss=f"{loss_value:.4f}")
					
					if config.training.log_every > 0 and global_step % config.training.log_every == 0:
						write_jsonl_log(metrics_log, {
							"event": "train", "step": global_step,
							"epoch": epoch + 1, "loss": loss_value,
							"lr": lr_scheduler.get_last_lr()[0],
						})
					
					if (global_step == total_training_steps) or (config.training.save_every > 0 and global_step % config.training.save_every == 0):
						save_checkpoint_with_evaluation(
							accelerator, model, eval_dataloader,
							config, global_step, output_dir
						)
						write_jsonl_log(metrics_log, {
							"event": "eval", "step": global_step
						})

	progress_bar.close()
	write_jsonl_log(metrics_log, {"event": "end"})
	if metrics_log is not None:
		metrics_log.close()
	accelerator.end_training()


if __name__ == "__main__":
	main()
