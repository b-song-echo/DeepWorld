import torch
import torch.nn as nn
from torch import Tensor

from src.config import WorldModelConfig
from src.models.qwen_brain import QwenBrain
from src.models.wan_renderer import WanRenderer
from src.utils.compat import load_diffusers_classes


def build_wan_scheduler(config: WorldModelConfig, scheduler_cls):
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
			scheduler = scheduler_cls.from_pretrained(config.wan_renderer.checkpoint_path, **load_kwargs)
		elif hasattr(scheduler_cls, "load_config"):
			scheduler_config = scheduler_cls.load_config(config.wan_renderer.checkpoint_path, **load_kwargs)
			scheduler = scheduler_cls.from_config(scheduler_config)
		else:
			scheduler = scheduler_cls()
	except Exception as error:
		import warnings
		warnings.warn(
			f"Failed to load Wan scheduler config from {config.wan_renderer.checkpoint_path!r}: {error}. "
			"Falling back to the diffusers default scheduler.",
			RuntimeWarning,
		)
		scheduler = scheduler_cls()

	if config.wan_renderer.train_scheduler_steps is None:
		return scheduler

	train_scheduler_steps = int(config.wan_renderer.train_scheduler_steps)
	if int(scheduler.config.num_train_timesteps) == train_scheduler_steps:
		return scheduler

	return scheduler_cls.from_config(scheduler.config, num_train_timesteps=train_scheduler_steps)


class DeepWorld(nn.Module):
	"""Top-level grounded video model combining a Qwen brain and Wan renderer.

	Args:
		config: Root configuration object describing all model and training settings.
	"""

	def __init__(self, config: WorldModelConfig):
		super().__init__()
		self.config = config
		self.brain = QwenBrain(
			config.qwen_brain,
			gradient_checkpointing=config.training.gradient_checkpointing,
		)
		self.renderer = WanRenderer(
			config.wan_renderer,
			condition_dim=self.brain.hidden_size,
			gradient_checkpointing=config.training.gradient_checkpointing,
		)
		_, FlowMatchEulerDiscreteScheduler, _ = load_diffusers_classes()
		self.scheduler = build_wan_scheduler(config, FlowMatchEulerDiscreteScheduler)

	def _build_loss_masks(self, frame_counts: Tensor, latents: Tensor) -> tuple[Tensor, Tensor]:
		"""Create latent-space and token-space validity masks for padded batches.

		Args:
			frame_counts: Valid video frame count per batch element before padding.
			latents: Encoded Wan latents for the padded batch.

		Returns:
			A tuple containing:
			- `latent_mask` with shape `(B, 1, T_lat, H_lat, W_lat)`,
			- `token_mask` with shape `(B, N_tokens)`.
		"""

		device = latents.device
		batch_size, _, latent_frames, latent_height, latent_width = latents.size()
		valid_latent_frames = ((frame_counts.to(device, non_blocking=True) - 1) // self.renderer.vae.config.scale_factor_temporal) + 1

		latent_mask = torch.zeros(batch_size, 1, latent_frames, latent_height, latent_width, device=device, dtype=latents.dtype)
		for batch_index, valid_frames in enumerate(valid_latent_frames.tolist()):
			latent_mask[batch_index, :, :valid_frames] = 1.0

		p_t, p_h, p_w = self.renderer.transformer.config.patch_size
		token_frames = latent_frames // p_t
		token_height = latent_height // p_h
		token_width = latent_width // p_w
		token_mask = torch.zeros(batch_size, token_frames * token_height * token_width, device=device, dtype=torch.bool)
		for batch_index, valid_frames in enumerate(valid_latent_frames.tolist()):
			valid_tokens = (valid_frames // p_t) * token_height * token_width
			token_mask[batch_index, :valid_tokens] = True

		return latent_mask, token_mask

	def _expand_renderer_training_batch(
		self,
		latents: Tensor,
		latent_mask: Tensor,
		token_mask: Tensor,
		condition_hidden_states: Tensor,
	) -> tuple[Tensor, Tensor, Tensor, Tensor]:
		"""Duplicate renderer-side inputs for independent diffusion timestep training.

		The Qwen brain runs once per real sample. The lightweight Wan renderer can
		then see multiple noisy versions of each encoded latent by repeating the
		clean latents and conditioning states along the batch dimension.

		Args:
			latents: Clean Wan latents with shape `(B, C, T, H, W)`.
			latent_mask: Latent loss mask aligned with `latents`.
			token_mask: Wan patch-token validity mask aligned with `latents`.
			condition_hidden_states: Qwen generation states aligned with Wan tokens.

		Returns:
			The repeated latents, latent masks, token masks, and conditioning states.
		"""

		multiplier = self.config.training.renderer_batch_multiplier
		if multiplier == 1:
			return latents, latent_mask, token_mask, condition_hidden_states

		return (
			latents.repeat_interleave(multiplier, dim=0),
			latent_mask.repeat_interleave(multiplier, dim=0),
			token_mask.repeat_interleave(multiplier, dim=0),
			condition_hidden_states.repeat_interleave(multiplier, dim=0),
		)

	def _apply_condition_dropout(self, token_mask: Tensor) -> Tensor:
		"""Randomly replace full renderer samples with the learned null condition.

		The Wan renderer already interprets `False` token-mask entries as null
		conditioning slots. Clearing an entire row therefore drops all Qwen
		conditioning for that renderer-side sample while preserving tensor shapes.

		Args:
			token_mask: Boolean conditioning validity mask with shape `(B, N)`.

		Returns:
			A mask with some complete rows cleared during training.
		"""

		dropout_prob = self.config.wan_renderer.condition_dropout_prob
		if not self.training or dropout_prob <= 0.0:
			return token_mask
		if dropout_prob >= 1.0:
			return torch.zeros_like(token_mask)

		keep_condition = torch.rand(token_mask.size(0), device=token_mask.device) >= dropout_prob
		return token_mask & keep_condition.unsqueeze(1)

	def forward(
		self,
		batch: dict[str, Tensor],
		return_auxiliary: bool = False,
		generate_samples: bool = False,
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

		if generate_samples:
			return {"videos": self.generate(batch, generator=generator)}

		videos = batch["videos"].to(device=next(self.parameters()).device, non_blocking=True)
		latents = self.renderer.encode_videos(videos, sample_posterior=self.config.wan_renderer.vae_sample_posterior)
		latent_mask, token_mask = self._build_loss_masks(batch["video_frame_counts"], latents)
		latent_patch_grids = self.renderer.latent_patch_grids(latents)

		brain_outputs = self.brain(batch, latent_patch_grids=latent_patch_grids)
		condition_hidden_states = brain_outputs["gen_hidden_states"]

		latents, latent_mask, token_mask, condition_hidden_states = self._expand_renderer_training_batch(
			latents, latent_mask, token_mask, condition_hidden_states,
		)

		noise = torch.randn_like(latents)
		sigmas = torch.rand(latents.size(0), device=latents.device, dtype=latents.dtype)
		sigma_view = sigmas.view(-1, 1, 1, 1, 1)
		noisy_latents = sigma_view * noise + (1.0 - sigma_view) * latents
		target = noise - latents

		model_output = self.renderer(
			hidden_states=noisy_latents,
			timestep=sigmas * self.scheduler.config.num_train_timesteps,
			condition_hidden_states=condition_hidden_states,
			token_mask=self._apply_condition_dropout(token_mask),
			return_dict=True,
		)["sample"]

		loss = (model_output.float() - target.float()).pow(2)
		loss_mask = latent_mask.float()
		loss_per_sample = (loss * loss_mask).flatten(1).sum(1)
		valid_per_sample = loss_mask.flatten(1).sum(1) * loss.size(1)
		loss = (loss_per_sample / valid_per_sample.clamp_min(1.0)).mean()

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
		batch: dict[str, Tensor],
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
			batch["txt_input_ids"].size(0),
			self.renderer.transformer.config.in_channels,
			latent_frames,
			height // self.renderer.vae.config.scale_factor_spatial,
			width // self.renderer.vae.config.scale_factor_spatial,
			device=device,
			dtype=self.renderer.transformer.patch_embedding.weight.dtype,
			generator=generator,
		)

		latent_patch_grids = self.renderer.latent_patch_grids(latents)
		brain_outputs = self.brain(batch, latent_patch_grids=latent_patch_grids)
		token_mask = brain_outputs["gen_mask"][:, : brain_outputs["gen_hidden_states"].size(1)]
		self.scheduler.set_timesteps(num_inference_steps or self.config.wan_renderer.inference_steps, device=device)

		for timestep in self.scheduler.timesteps:
			model_output = self.renderer(
				hidden_states=latents,
				timestep=timestep.expand(latents.size(0)),
				condition_hidden_states=brain_outputs["gen_hidden_states"],
				token_mask=token_mask,
				return_dict=True,
			)["sample"]
			latents = self.scheduler.step(model_output, timestep, latents, return_dict=False)[0]

		return self.renderer.decode_latents(latents)
