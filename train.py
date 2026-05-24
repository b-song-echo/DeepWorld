import os
import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import TextIO

import torch
import torch.nn as nn
import yaml
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import enable_fsdp_ram_efficient_loading, set_seed
from huggingface_hub import load_torch_model, save_torch_state_dict
from torch import Tensor
from torch.distributed.fsdp import FullStateDictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoProcessor, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from src.config import load_config, WorldModelConfig
from src.data import WorldModelBatchCollator, build_dataset
from src.models import DeepWorld
from src.utils import get_world_size
from src.utils.video import save_image_tensor, save_video_tensor


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


def create_accelerator(
	output_dir: str,
	grad_acc_steps: int,
	mixed_precision: str,
	use_fsdp: bool,
) -> Accelerator:
	"""Create the Accelerate runtime and optional in-code FSDP plugin.

	Args:
		output_dir: Directory used for tracker output.
		grad_acc_steps: Gradient accumulation steps.
		mixed_precision: Mixed precision mode such as `bf16`.
		use_fsdp: Whether multi-process training should use FSDP instead of DDP.

	Returns:
		A configured `Accelerator`.
	"""

	world_size = get_world_size()
	fsdp_enabled = use_fsdp and world_size > 1
	enable_fsdp_ram_efficient_loading()

	fsdp_plugin = FullyShardedDataParallelPlugin(
		fsdp_version=1,
		sharding_strategy="FULL_SHARD",
		auto_wrap_policy="TRANSFORMER_BASED_WRAP",
		transformer_cls_names_to_wrap=["MoFfnQwenDecoderLayer"],
		state_dict_type="FULL_STATE_DICT",
		state_dict_config=FullStateDictConfig(
			offload_to_cpu=True, rank0_only=True
		),
		backward_prefetch="BACKWARD_PRE",
		forward_prefetch=False,
		activation_checkpointing=False,
		ignored_modules=None,
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


def configure_fsdp_model(accelerator: Accelerator, model: DeepWorld) -> None:
	"""Attach model-specific FSDP settings after RAM-efficient construction.

	Accelerate must be created before Hugging Face `from_pretrained` calls so
	only rank 0 materializes checkpoint tensors. The ignored-module list depends
	on the constructed model, so it is filled in afterward before `prepare`.

	Args:
		accelerator: Active Accelerate runtime.
		model: Unprepared world-model instance.
	"""

	fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
	if fsdp_plugin is None:
		return

	# FSDP flattens wrapped parameters, so every non-ignored branch must share
	# one dtype. Frozen encoders keep their checkpoint dtypes and are ignored.
	target_dtype = next(model.brain.language_model.parameters()).dtype
	managed_mods = (
		model.brain.language_model,
		model.brain.geo_bridge,
		model.brain.segment_tokens,
		model.renderer.transformer,
		model.renderer.condition_proj,
	)
	managed_params = (
		model.brain.gen_slot_token,
		model.renderer.null_cond,
	)
	managed_dtypes = {
		param.dtype
		for mod in managed_mods
		for param in mod.parameters(recurse=True)
	}
	managed_dtypes.update(param.dtype for param in managed_params)
	if managed_dtypes - {target_dtype}:
		for mod in managed_mods:
			mod.to(dtype=target_dtype)
		for param in managed_params:
			with torch.no_grad():
				param.data = param.data.to(dtype=target_dtype)

	fsdp_plugin.ignored_modules = [
		model.renderer.vae,
		model.brain.geometry_encoder,
		model.brain.vision_encoder,
	]


def load_pretrained_model_state(
	model: DeepWorld,
	pretrained_model_path: str | None,
	accelerator: Accelerator,
) -> None:
	"""Load an optional full-model state dict without multiplying CPU RAM use.

	Only the main process reads `model.pt`; distributed wrappers then broadcast
	rank 0's module state during `accelerator.prepare`.

	Args:
		model: Unprepared world-model instance.
		pretrained_model_path: Optional path to a saved full-model state dict.
		accelerator: Active Accelerate runtime.
	"""

	if pretrained_model_path is not None and accelerator.is_main_process:
		load_result = load_torch_model(
			model, Path(pretrained_model_path),
			strict=False, safe=True, weights_only=True,
			map_location="cpu", mmap=True,
		)
		if load_result.missing_keys or load_result.unexpected_keys:
			raise RuntimeError(
				"Pretrained checkpoint does not match the model state dict: "
				f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}."
			)
	accelerator.wait_for_everyone()


def log_param_summary(accelerator: Accelerator, model: DeepWorld) -> None:
	"""Print a compact parameter and static-weight summary for memory debugging."""

	def count_params(mod: nn.Module, trainable_only: bool = False) -> tuple[int, int]:
		num_params = 0
		num_bytes = 0
		for param in mod.parameters():
			if trainable_only and not param.requires_grad:
				continue
			num_params += param.numel()
			num_bytes += param.numel() * param.element_size()
		return num_params, num_bytes

	def format_count(value: int) -> str:
		for unit in ["", "K", "M", "B", "T"]:
			if abs(value) < 1000 or unit == "T":
				return f"{value:.2f}{unit}" if unit else str(value)
			value /= 1000
		return str(value)

	def format_bytes(value: int) -> str:
		for unit in ["B", "KB", "MB", "GB", "TB"]:
			if abs(value) < 1024 or unit == "TB":
				return f"{value:.2f}{unit}" if unit != "B" else f"{value}B"
			value /= 1024
		return f"{value:.2f}TB"

	total_params, total_bytes = count_params(model)
	trainable_params, trainable_bytes = count_params(model, trainable_only=True)
	accelerator.print(
		"Model parameters: "
		f"total={format_count(total_params)} ({format_bytes(total_bytes)}), "
		f"trainable={format_count(trainable_params)} ({format_bytes(trainable_bytes)})."
	)
	for name, mod in (
		("brain.language_model", model.brain.language_model),
		("brain.vision_encoder", model.brain.vision_encoder),
		("brain.geometry_encoder", model.brain.geometry_encoder),
		("renderer.transformer", model.renderer.transformer),
		("renderer.vae", model.renderer.vae),
	):
		mod_params, mod_bytes = count_params(mod)
		mod_trainable, _ = count_params(mod, trainable_only=True)
		accelerator.print(
			f"  {name}: {format_count(mod_params)} params, {format_bytes(mod_bytes)}, "
			f"trainable={format_count(mod_trainable)}"
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

	def train_params(mod: nn.Module) -> list[nn.Parameter]:
		return [p for p in mod.parameters() if p.requires_grad]

	def train_params_except(
		mod: nn.Module,
		excluded_params: list[nn.Parameter]
	) -> list[nn.Parameter]:
		excluded_ids = {id(p) for p in excluded_params}
		return [p for p in train_params(mod) if id(p) not in excluded_ids]

	def add_optimizer_group(
		param_groups: list[dict],
		group_name: str,
		params: list[nn.Parameter],
		lr: float,
	) -> None:
		if len(params) == 0:
			return
		param_groups.append({
			"name": group_name,
			"params": params,
			"lr": lr,
		})

	brain_language_model_params = train_params(model.brain.language_model)
	brain_others_params = train_params_except(model.brain, brain_language_model_params)

	renderer_transformer_params = train_params(model.renderer.transformer)
	renderer_others_params = train_params_except(model.renderer, renderer_transformer_params)

	grouped_ids = {
		id(param)
		for param in (
			brain_language_model_params
			+ brain_others_params
			+ renderer_transformer_params
			+ renderer_others_params
		)
	}
	remainder_params = [param for param in model.parameters() if param.requires_grad and id(param) not in grouped_ids]

	brain_language_model_lr = config.optimizer.brain_language_model_learning_rate or config.optimizer.learning_rate
	brain_others_lr = config.optimizer.brain_others_learning_rate or brain_language_model_lr
	renderer_transformer_lr = config.optimizer.renderer_transformer_learning_rate or config.optimizer.learning_rate
	renderer_others_lr = config.optimizer.renderer_others_learning_rate or renderer_transformer_lr

	param_groups: list[dict] = []
	add_optimizer_group(
		param_groups, "brain_language_model",
		brain_language_model_params,
		brain_language_model_lr,
	)
	add_optimizer_group(
		param_groups, "brain_others",
		brain_others_params,
		brain_others_lr,
	)
	add_optimizer_group(
		param_groups, "renderer_transformer",
		renderer_transformer_params,
		renderer_transformer_lr,
	)
	add_optimizer_group(
		param_groups, "renderer_others",
		renderer_others_params,
		renderer_others_lr,
	)
	add_optimizer_group(
		param_groups, "remainder",
		remainder_params,
		config.optimizer.learning_rate,
	)
	if len(param_groups) == 0:
		raise RuntimeError("No trainable parameters were found for optimizer construction.")

	return AdamW(
		param_groups,
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
	"""Build a deterministic per-rank evaluation loader over the validation split.

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
	if not config.dataset.eval_manifest_path:
		raise ValueError("`dataset.eval_manifest_path` must be set when `training.eval_num_samples` is positive.")
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
	eval_dataset_config = replace(
		config.dataset,
		manifest_path=config.dataset.eval_manifest_path,
		num_samples=eval_num_samples,
		shuffle=False,
	)
	eval_dataset = build_dataset(eval_dataset_config)
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


def evaluate(
	accelerator: Accelerator,
	model: nn.Module,
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

	def save_sample_bundle(
		batch: dict,
		generated_video: Tensor,
		batch_index: int,
		global_eval_index: int,
	) -> None:
		sample_dir = checkpoint_dir / f"sample_{global_eval_index:04d}"
		sample_dir.mkdir(parents=True, exist_ok=True)
		save_video_tensor(generated_video, sample_dir / "generated.mp4", duration_seconds=config.dataset.video_duration)
		save_video_tensor(batch["videos"][batch_index], sample_dir / "ground_truth.mp4", duration_seconds=config.dataset.video_duration)

		prompt = batch["prompts"][batch_index]
		(sample_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

		reference_dir = sample_dir / "references"
		reference_mask = batch["reference_mask"][batch_index]
		reference_images = batch["geo_images"][batch_index]
		for ref_index, is_valid in enumerate(reference_mask.tolist()):
			if not is_valid:
				continue
			save_image_tensor(reference_images[ref_index], reference_dir / f"reference_{ref_index:02d}.png")

	per_process_samples = config.training.eval_num_samples // accelerator.num_processes
	was_training = model.training
	model.eval()
	num_saved = 0
	try:
		for batch in eval_dataloader:
			if num_saved >= per_process_samples:
				break

			global_eval_index = num_saved * accelerator.num_processes + accelerator.process_index
			generator = torch.Generator(device=accelerator.device)
			generator = generator.manual_seed(config.training.seed + step * 1000003 + global_eval_index)
			
			with torch.no_grad():
				with accelerator.autocast():
					outputs = model(batch, generate_samples=True, generator=generator)

			videos = outputs["videos"].detach().cpu()
			for batch_index in range(videos.size(0)):
				if num_saved >= per_process_samples:
					break
				global_eval_index = num_saved * accelerator.num_processes + accelerator.process_index
				save_sample_bundle(
					batch=batch,
					generated_video=videos[batch_index],
					batch_index=batch_index,
					global_eval_index=global_eval_index,
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


def save_checkpoint_with_evaluation(
	accelerator: Accelerator,
	model: nn.Module,
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

	checkpoint_dir = output_dir / f"checkpoint-{step}"
	if accelerator.is_main_process:
		checkpoint_dir.mkdir(parents=True, exist_ok=True)
	accelerator.wait_for_everyone()
	
	evaluate(accelerator, model, eval_dataloader, config, step, checkpoint_dir)
	
	# NOTE: `get_state_dict` is on all ranks.
	accelerator.wait_for_everyone()
	state_dict = accelerator.get_state_dict(model)
	if accelerator.is_main_process:
		checkpoint_dir.mkdir(parents=True, exist_ok=True)
		save_torch_state_dict(
			state_dict, checkpoint_dir,
			max_shard_size="5GB",
			metadata={"format": "pt"},
			safe_serialization=True,
		)
	accelerator.wait_for_everyone()


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

	accelerator = create_accelerator(
		output_dir=config.training.output_dir,
		grad_acc_steps=config.training.gradient_accumulation_steps,
		mixed_precision=config.training.mixed_precision,
		use_fsdp=config.training.use_fsdp,
	)
	model = DeepWorld(config)
	load_pretrained_model_state(model, config.training.pretrained_model_path, accelerator)
	configure_fsdp_model(accelerator, model)

	save_config_snapshot(config, output_dir, accelerator)
	metrics_log = open_jsonl_log(output_dir, accelerator)
	write_jsonl_log(metrics_log, {"event": "start"})
	log_param_summary(accelerator, model)

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
		shuffle=config.dataset.shuffle,
		dataset_config=config.dataset,
		collator=collator,
	)
	eval_dataloader = build_eval_dataloader(config, collator, accelerator)

	model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

	grad_clip_params = tuple(p for group in optimizer.param_groups for p in group["params"])

	steps_per_epoch = math.ceil(len(dataloader) / config.training.gradient_accumulation_steps)
	max_epoch_steps = (
		steps_per_epoch * config.training.max_total_epochs
		if config.training.max_total_epochs is not None else config.training.max_total_steps
	)
	max_total_steps = config.training.max_total_steps or max_epoch_steps
	total_training_steps = min(max_epoch_steps, max_total_steps)
	
	lr_scheduler = build_lr_scheduler(optimizer, config, total_training_steps)
	progress_bar = tqdm(
		total=total_training_steps,
		desc="Training",
		disable=not accelerator.is_local_main_process,
		dynamic_ncols=True,
	)
	progress_bar.set_postfix(step=0, epoch=0, loss="n/a")

	global_step = 0
	epoch_index = 0
	accumulated_losses: list[Tensor] = []
	while global_step < total_training_steps:
		model.train()
		for batch in dataloader:
			with accelerator.accumulate(model):
				with accelerator.autocast():
					loss = model(batch)["loss"]
				accumulated_losses.append(loss.detach().float())
				accelerator.backward(loss)
				if accelerator.sync_gradients:
					accelerator.clip_grad_norm_(grad_clip_params, config.optimizer.max_grad_norm)
				optimizer.step()
				optimizer.zero_grad(set_to_none=True)

				if accelerator.sync_gradients:
					lr_scheduler.step()
					global_step += 1
					macro_loss = torch.stack(accumulated_losses).mean()
					accumulated_losses.clear()
					loss_value = accelerator.gather(macro_loss).mean().item()
					
					progress_bar.update(1)
					progress_bar.set_postfix(step=global_step, epoch=epoch_index + 1, loss=f"{loss_value:.4f}")
					
					if config.training.log_every > 0 and global_step % config.training.log_every == 0:
						write_jsonl_log(metrics_log, {
							"event": "train", "step": global_step,
							"epoch": epoch_index + 1, "loss": loss_value,
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
					if global_step >= total_training_steps:
						break
		epoch_index += 1

	progress_bar.close()
	write_jsonl_log(metrics_log, {"event": "end"})
	if metrics_log is not None:
		metrics_log.close()
	accelerator.end_training()


if __name__ == "__main__":
	main()
