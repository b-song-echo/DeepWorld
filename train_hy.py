import argparse
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
	CheckpointImpl,
	apply_activation_checkpointing,
	checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm.auto import tqdm
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from hyvideo.hyvideo.commons.parallel_states import get_parallel_state, initialize_parallel_state
from src.config import DeepWorldHYConfig, load_hy_config
from src.data import DeepWorldHYBatchCollator, build_dataset
from src.deep_world_hy import MMTripleStreamBlock, MMSingleStreamBlock, DeepWorldHY
from src.utils import (
	configure_offline_runtime,
	get_world_size,
	open_rank0_jsonl_log,
	save_config_yaml,
	save_image_tensor,
	save_video_tensor,
	seed_python_and_torch,
	write_jsonl_log,
)


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for DeepWorldHY training."""

	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
	return parser.parse_args()


def set_runtime_environment() -> None:
	"""Apply process-wide runtime defaults before model construction."""

	configure_offline_runtime()


def initialize_torchrun(config: DeepWorldHYConfig) -> tuple[torch.device, int, int, bool]:
	"""Initialize torch distributed state and Hunyuan parallel meshes."""

	if "RANK" in os.environ:
		rank = int(os.environ["RANK"])
		world_size = int(os.environ.get("WORLD_SIZE", "1"))
		local_rank = int(os.environ.get("LOCAL_RANK", "0"))
		if torch.cuda.is_available():
			torch.cuda.set_device(local_rank)
			device = torch.device(f"cuda:{local_rank}")
			dist.init_process_group("nccl")
		else:
			device = torch.device("cpu")
			dist.init_process_group("gloo")
	else:
		rank = 0
		world_size = 1
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if config.training.sequence_parallel_size > world_size:
		raise ValueError(
			"`training.sequence_parallel_size` cannot exceed world size; "
			f"got sp={config.training.sequence_parallel_size}, world_size={world_size}."
		)
	if world_size % config.training.sequence_parallel_size != 0:
		raise ValueError(
			"`training.sequence_parallel_size` must divide world size; "
			f"got sp={config.training.sequence_parallel_size}, world_size={world_size}."
		)

	if torch.cuda.is_available():
		initialize_parallel_state(
			sp=config.training.sequence_parallel_size,
			dp_replicate=config.training.data_parallel_replicate,
		)
	is_main_process = rank == 0
	return device, rank, world_size, is_main_process


def set_process_seed(seed: int) -> None:
	"""Seed Python and torch RNGs for this process."""

	seed_python_and_torch(seed)


def save_config_snapshot(config: DeepWorldHYConfig, output_dir: Path, is_main_process: bool) -> None:
	"""Write the resolved config beside logs and checkpoints."""

	if is_main_process:
		save_config_yaml(config, output_dir)
	if dist.is_available() and dist.is_initialized():
		dist.barrier()


def open_metrics_log(output_dir: Path, is_main_process: bool):
	"""Open the rank-0 JSONL metrics log."""

	return open_rank0_jsonl_log(output_dir, is_main_process)


def get_data_parallel_rank_size(rank: int) -> tuple[int, int]:
	"""Return the rank and size of the data-parallel mesh."""

	parallel_state = get_parallel_state() if torch.cuda.is_available() else None
	if parallel_state is None or get_world_size() <= 1:
		return rank, 1
	return parallel_state.world_mesh["dp"].get_local_rank(), parallel_state.world_mesh["dp"].size()


def is_sequence_parallel_writer() -> bool:
	"""Return whether this process should write rank-local evaluation artifacts."""

	parallel_state = get_parallel_state() if torch.cuda.is_available() else None
	return parallel_state is None or not getattr(parallel_state, "sp_enabled", False) or parallel_state.sp_rank == 0


# TODO: The following three dataloader related functions are a bit messy, and should be refactored. Why one called `create_dataloader` while the other didn't? Besides, make the name clearer: `build_train_dataloader` and `build_eval_dataloader`.
def create_deepworld_hy_dataloader(
	config: DeepWorldHYConfig,
	collator: DeepWorldHYBatchCollator,
) -> DataLoader:
	"""Build a dataloader whose SP ranks receive identical samples."""

	dataset = build_dataset(config.dataset)
	parallel_state = get_parallel_state() if torch.cuda.is_available() else None
	if parallel_state is not None and get_world_size() > 1:
		dp_rank, dp_size = get_data_parallel_rank_size(rank=0)
		sampler = DistributedSampler(
			dataset,
			num_replicas=dp_size,
			rank=dp_rank,
			shuffle=config.dataset.shuffle,
			drop_last=False,
		)
		shuffle = False
	else:
		sampler = None
		shuffle = config.dataset.shuffle

	loader_kwargs = {
		"dataset": dataset,
		"batch_size": None,
		"shuffle": shuffle,
		"sampler": sampler,
		"num_workers": config.dataset.num_workers,
		"pin_memory": config.dataset.pin_memory and torch.cuda.is_available(),
		"persistent_workers": config.dataset.persistent_workers and config.dataset.num_workers > 0,
		"collate_fn": collator,
	}
	if config.dataset.num_workers > 0 and config.dataset.prefetch_factor is not None:
		loader_kwargs["prefetch_factor"] = config.dataset.prefetch_factor
	return DataLoader(**loader_kwargs)


def create_dataloader(
	dataset,
	dataset_config,
	collator: DeepWorldHYBatchCollator,
	shuffle: bool,
) -> DataLoader:
	"""Build a basic dataloader for pre-partitioned HY datasets."""

	loader_kwargs = {
		"dataset": dataset,
		"batch_size": None,
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
	config: DeepWorldHYConfig,
	collator: DeepWorldHYBatchCollator,
	rank: int,
) -> tuple[DataLoader, list[int]] | None:
	"""Build a deterministic eval loader partitioned across data-parallel groups."""

	if config.training.eval_every == 0 or config.training.eval_num_samples == 0:
		return None
	if not config.dataset.eval_manifest_path:
		raise ValueError("`dataset.eval_manifest_path` must be set when HY evaluation is enabled.")

	dp_rank, dp_size = get_data_parallel_rank_size(rank)
	eval_dataset_config = replace(
		config.dataset,
		manifest_path=config.dataset.eval_manifest_path,
		num_samples=config.training.eval_num_samples,
		shuffle=False,
	)
	eval_dataset = build_dataset(eval_dataset_config)
	total_eval_samples = len(eval_dataset)
	rank_indices = list(range(dp_rank, total_eval_samples, dp_size))
	eval_source = Subset(eval_dataset, rank_indices)
	return create_dataloader(
		dataset=eval_source,
		dataset_config=eval_dataset_config,
		collator=collator,
		shuffle=False,
	), rank_indices


def evaluate(
	model: DeepWorldHY,
	eval_bundle: tuple[DataLoader, list[int]] | None,
	config: DeepWorldHYConfig,
	step: int,
	output_dir: Path,
	device: torch.device,
) -> None:
	"""Generate rank-local HY evaluation videos and save them to disk."""

	if eval_bundle is None:
		return

	eval_dataloader, rank_indices = eval_bundle
	eval_dir = output_dir / "evaluation" / f"step_{step:08d}"
	can_write = is_sequence_parallel_writer()
	if can_write:
		eval_dir.mkdir(parents=True, exist_ok=True)
	if dist.is_available() and dist.is_initialized():
		dist.barrier()

	def save_sample_bundle(batch: dict[str, Any], generated_video: torch.Tensor, eval_index: int) -> None:
		"""Save generated output, target video, prompt, and reference images."""

		sample_dir = eval_dir / f"sample_{eval_index:04d}"
		sample_dir.mkdir(parents=True, exist_ok=True)
		save_video_tensor(generated_video, sample_dir / "generated.mp4", duration_seconds=config.dataset.video_duration)
		save_video_tensor(batch["video"], sample_dir / "ground_truth.mp4", duration_seconds=config.dataset.video_duration)
		(sample_dir / "prompt.txt").write_text(batch["prompt"] + "\n", encoding="utf-8")

		reference_dir = sample_dir / "references"
		reference_images = batch["vis_ref_images"]
		for ref_index in range(reference_images.size(0)):
			save_image_tensor(reference_images[ref_index], reference_dir / f"reference_{ref_index:02d}.png")

	was_training = model.training
	model.eval()
	num_seen = 0
	try:
		for batch in eval_dataloader:
			if num_seen >= len(rank_indices):
				break
			eval_index = rank_indices[num_seen]
			generator = torch.Generator(device=device)
			generator.manual_seed(config.training.seed + step * 1000003 + eval_index)
			with torch.no_grad():
				with torch.autocast(
					device_type="cuda",
					dtype=torch.bfloat16,
					enabled=torch.cuda.is_available() and config.training.mixed_precision == "bf16",
				):
					outputs = model(batch, generate_samples=True, generator=generator)

			video = outputs["video"].detach().cpu()
			if can_write:
				save_sample_bundle(batch, video, eval_index)
			num_seen += 1
	finally:
		if was_training:
			model.train()

	if num_seen != len(rank_indices):
		raise RuntimeError(
			"Evaluation dataloader ended before this data-parallel rank generated its assigned samples; "
			f"saved={num_seen}, expected={len(rank_indices)}."
		)
	if dist.is_available() and dist.is_initialized():
		dist.barrier()


def apply_gradient_checkpointing(model: DeepWorldHY) -> None:
	"""Checkpoint Hunyuan transformer blocks with a non-reentrant wrapper."""

	block_types = (MMTripleStreamBlock, MMSingleStreamBlock)

	def non_reentrant_wrapper(module: nn.Module) -> nn.Module:
		return checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

	def should_checkpoint(module: nn.Module) -> bool:
		return isinstance(module, block_types)

	apply_activation_checkpointing(
		model.transformer,
		checkpoint_wrapper_fn=non_reentrant_wrapper,
		check_fn=should_checkpoint,
	)


def apply_fsdp2(model: DeepWorldHY, config: DeepWorldHYConfig, is_main_process: bool) -> None:
	"""Apply Hunyuan-style FSDP2 sharding to trainable modules."""

	if not config.training.use_fsdp or get_world_size() <= 1:
		return

	param_dtype = torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float32
	mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
	fsdp_kwargs: dict[str, Any] = {"mp_policy": mp_policy}
	if torch.cuda.is_available():
		try:
			fsdp_kwargs["mesh"] = get_parallel_state().fsdp_mesh
		except Exception as error:
			if is_main_process:
				print(f"Could not attach Hunyuan FSDP mesh, falling back to default process group: {error}")

	for block in list(model.transformer.double_blocks) + list(model.transformer.single_blocks):
		fully_shard(block, **fsdp_kwargs)
	fully_shard(model.transformer, **fsdp_kwargs)
	fully_shard(model.geo_in, **fsdp_kwargs)
	fully_shard(model.modality_embeddings, **fsdp_kwargs)
	if getattr(model, "vis_in", None) is not getattr(model.transformer, "vision_in", None):
		fully_shard(model.vis_in, **fsdp_kwargs)


def build_optimizer(model: DeepWorldHY, config: DeepWorldHYConfig) -> AdamW:
	"""Build AdamW groups for adapters, geometry stream, and projectors."""

	def add_group(groups: list[dict], name: str, params: list[nn.Parameter], lr: float) -> None:
		if params:
			groups.append({"name": name, "params": params, "lr": lr})

	projector_prefixes = ("geo_in.", "vis_in.", "modality_embeddings.")
	geometry_params: list[nn.Parameter] = []
	adapter_params: list[nn.Parameter] = []
	projector_params: list[nn.Parameter] = []
	remainder_params: list[nn.Parameter] = []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if ".geo." in name:
			geometry_params.append(param)
		elif name.startswith(projector_prefixes):
			projector_params.append(param)
		elif name.startswith("transformer."):
			adapter_params.append(param)
		else:
			remainder_params.append(param)

	transformer_lr = config.optimizer.adapter_learning_rate or config.optimizer.learning_rate
	geometry_lr = config.optimizer.geo_stream_learning_rate or config.optimizer.learning_rate
	projector_lr = config.optimizer.projector_learning_rate or geometry_lr

	param_groups: list[dict] = []
	add_group(param_groups, "hy_transformer_adapters", adapter_params, transformer_lr)
	add_group(param_groups, "hy_geometry_stream", geometry_params, geometry_lr)
	add_group(param_groups, "hy_projectors", projector_params, projector_lr)
	add_group(param_groups, "remainder", remainder_params, config.optimizer.learning_rate)
	if not param_groups:
		raise RuntimeError("No trainable DeepWorldHY parameters were found.")

	return AdamW(
		param_groups,
		weight_decay=config.optimizer.weight_decay,
		betas=tuple(config.optimizer.betas),
		eps=config.optimizer.eps,
	)


def build_lr_scheduler(optimizer: AdamW, config: DeepWorldHYConfig, total_training_steps: int):
	"""Create the configured warmup scheduler."""

	if config.optimizer.lr_schedule == "constant":
		return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config.optimizer.warmup_steps)
	if config.optimizer.lr_schedule == "cosine":
		return get_cosine_schedule_with_warmup(
			optimizer,
			num_warmup_steps=config.optimizer.warmup_steps,
			num_training_steps=total_training_steps,
		)
	raise ValueError(f"Unsupported optimizer.lr_schedule: {config.optimizer.lr_schedule!r}.")


def save_checkpoint(
	model: DeepWorldHY,
	optimizer: AdamW,
	lr_scheduler,
	step: int,
	output_dir: Path,
	is_main_process: bool,
) -> None:
	"""Save model, optimizer, and scheduler state with distributed checkpointing."""

	checkpoint_dir = output_dir / "checkpoints" / f"step_{step:08d}"
	transformer_dir = checkpoint_dir / "model"
	optimizer_dir = checkpoint_dir / "optimizer"
	if is_main_process:
		checkpoint_dir.mkdir(parents=True, exist_ok=True)
	if dist.is_available() and dist.is_initialized():
		dist.barrier()

	dcp.save({"model": get_model_state_dict(model)}, checkpoint_id=str(transformer_dir))
	dcp.save({"optimizer": get_optimizer_state_dict(model, optimizer)}, checkpoint_id=str(optimizer_dir))

	if is_main_process:
		torch.save({"global_step": step, "lr_scheduler": lr_scheduler.state_dict()}, checkpoint_dir / "training_state.pt")
	if dist.is_available() and dist.is_initialized():
		dist.barrier()


def mean_loss_across_ranks(loss: torch.Tensor) -> float:
	"""Return a scalar loss averaged across distributed ranks."""

	value = loss.detach().float()
	if dist.is_available() and dist.is_initialized():
		dist.all_reduce(value, op=dist.ReduceOp.AVG)
	return float(value.item())


def compute_total_training_steps(dataloader: DataLoader, config: DeepWorldHYConfig) -> int:
	"""Resolve the bounded number of optimizer steps for this run."""

	steps_per_epoch = math.ceil(len(dataloader) / config.training.gradient_accumulation_steps)
	max_epoch_steps = (
		steps_per_epoch * config.training.max_total_epochs
		if config.training.max_total_epochs is not None else config.training.max_total_steps
	)
	max_total_steps = config.training.max_total_steps or max_epoch_steps
	return min(max_epoch_steps, max_total_steps)


def main() -> None:
	"""Run DeepWorldHY training with torchrun, FSDP2, and Hunyuan parallel state."""

	set_runtime_environment()
	args = parse_args()
	config = load_hy_config(args.config)

	device, rank, _, is_main_process = initialize_torchrun(config)
	dp_rank, _ = get_data_parallel_rank_size(rank)
	set_process_seed(config.training.seed + dp_rank)

	output_dir = Path(config.training.output_dir)
	save_config_snapshot(config, output_dir, is_main_process)
	metrics_log = open_metrics_log(output_dir, is_main_process)
	write_jsonl_log(metrics_log, {"event": "start", "model": "DeepWorldHY"})

	model = DeepWorldHY(config).to(device)
	if config.training.gradient_checkpointing:
		apply_gradient_checkpointing(model)
	apply_fsdp2(model, config, is_main_process)
	optimizer = build_optimizer(model, config)

	collator = DeepWorldHYBatchCollator(config.dataset, geo_patch_size=model.geo_patch_size)
	dataloader = create_deepworld_hy_dataloader(config, collator)
	eval_bundle = build_eval_dataloader(config, collator, rank)
	total_training_steps = compute_total_training_steps(dataloader, config)
	lr_scheduler = build_lr_scheduler(optimizer, config, total_training_steps)

	progress_bar = tqdm(
		total=total_training_steps,
		desc="DeepWorldHY Training",
		disable=not is_main_process,
		dynamic_ncols=True,
	)
	global_step = 0
	epoch_index = 0
	accumulated_losses: list[torch.Tensor] = []
	model.train()
	while global_step < total_training_steps:
		sampler = getattr(dataloader, "sampler", None)
		if isinstance(sampler, DistributedSampler):
			sampler.set_epoch(epoch_index)

		for micro_step, batch in enumerate(dataloader):
			with torch.autocast(
				device_type="cuda",
				dtype=torch.bfloat16,
				enabled=torch.cuda.is_available() and config.training.mixed_precision == "bf16",
			):
				loss = model(batch)["loss"] / config.training.gradient_accumulation_steps
			loss.backward()
			accumulated_losses.append(loss.detach() * config.training.gradient_accumulation_steps)

			if (micro_step + 1) % config.training.gradient_accumulation_steps != 0:
				continue

			if config.optimizer.max_grad_norm > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad(set_to_none=True)

			global_step += 1
			macro_loss = torch.stack(accumulated_losses).mean()
			accumulated_losses.clear()
			loss_value = mean_loss_across_ranks(macro_loss)
			if is_main_process:
				progress_bar.update(1)
				progress_bar.set_postfix(step=global_step, epoch=epoch_index + 1, loss=f"{loss_value:.4f}")
			if config.training.log_every > 0 and global_step % config.training.log_every == 0:
				write_jsonl_log(metrics_log, {
					"event": "train",
					"step": global_step,
					"epoch": epoch_index + 1,
					"loss": loss_value,
					"lr": lr_scheduler.get_last_lr()[0],
				})
			if eval_bundle is not None and (
				global_step == total_training_steps
				or (config.training.eval_every > 0 and global_step % config.training.eval_every == 0)
			):
				evaluate(model, eval_bundle, config, global_step, output_dir, device)
				write_jsonl_log(metrics_log, {"event": "evaluation", "step": global_step})
			if (global_step == total_training_steps) or (
				config.training.save_every > 0 and global_step % config.training.save_every == 0
			):
				save_checkpoint(model, optimizer, lr_scheduler, global_step, output_dir, is_main_process)
				write_jsonl_log(metrics_log, {"event": "checkpoint", "step": global_step})
			if global_step >= total_training_steps:
				break
		epoch_index += 1

	if is_main_process:
		progress_bar.close()
	write_jsonl_log(metrics_log, {"event": "end", "step": global_step})
	if metrics_log is not None:
		metrics_log.close()
	if dist.is_available() and dist.is_initialized():
		dist.barrier()
		dist.destroy_process_group()


if __name__ == "__main__":
	main()
