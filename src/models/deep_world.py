import torch
import torch.nn as nn
from torch import Tensor

from src.config import WorldModelConfig
from src.models.qwen_brain import QwenBrain
from src.models.wan_renderer import WanRenderer
from src.utils.compat import load_diffusers_classes


def resolve_train_scheduler_steps(config: WorldModelConfig, scheduler_cls) -> int:
	"""Resolve the number of training diffusion timesteps for the Wan scheduler.

	Args:
		config: Root configuration object.
		scheduler_cls: Imported diffusers scheduler class.

	Returns:
		The training-time number of diffusion timesteps.
	"""

	if config.wan_renderer.train_scheduler_steps is not None:
		return int(config.wan_renderer.train_scheduler_steps)

	load_kwargs = {
		"subfolder": "scheduler",
		"local_files_only": True,
	}
	try:
		if hasattr(scheduler_cls, "from_pretrained"):
			scheduler = scheduler_cls.from_pretrained(config.wan_renderer.checkpoint_path, **load_kwargs)
			return int(scheduler.config.num_train_timesteps)
		if hasattr(scheduler_cls, "load_config"):
			scheduler_config = scheduler_cls.load_config(config.wan_renderer.checkpoint_path, **load_kwargs)
			return int(scheduler_config["num_train_timesteps"])
	except Exception as error:
		import warnings
		warnings.warn(
			f"Failed to infer Wan scheduler timesteps from {config.wan_renderer.checkpoint_path!r}: {error}. "
			"Falling back to 1000 training steps.",
			RuntimeWarning,
		)

	return 1000


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
		train_scheduler_steps = resolve_train_scheduler_steps(config, FlowMatchEulerDiscreteScheduler)
		self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=train_scheduler_steps)

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

	def forward(self, batch: dict[str, Tensor], return_auxiliary: bool = False) -> dict[str, Tensor]:
		"""Run one training forward pass and return latent-space denoising loss.

		Args:
			batch: Collated training batch produced by the dataset pipeline.
			return_auxiliary: Whether to include non-loss tensors in the return payload.

		Returns:
			A dictionary containing the scalar loss and, optionally, auxiliary tensors.
		"""

		videos = batch["videos"].to(device=next(self.parameters()).device, non_blocking=True)
		latents = self.renderer.encode_videos(videos, sample_posterior=True)
		latent_mask, token_mask = self._build_loss_masks(batch["video_frame_counts"], latents)
		latent_patch_grids = self.renderer.latent_patch_grids(latents)

		brain_outputs = self.brain(batch, latent_patch_grids=latent_patch_grids)

		noise = torch.randn_like(latents)
		sigmas = torch.rand(latents.size(0), device=latents.device, dtype=latents.dtype)
		sigma_view = sigmas.view(-1, 1, 1, 1, 1)
		noisy_latents = sigma_view * noise + (1.0 - sigma_view) * latents
		target = noise - latents
		timesteps = sigmas * self.scheduler.config.num_train_timesteps

		model_output = self.renderer(
			hidden_states=noisy_latents,
			timestep=timesteps,
			condition_hidden_states=brain_outputs["gen_hidden_states"],
			token_mask=token_mask,
			return_dict=True,
		)["sample"]

		loss = (model_output.float() - target.float()).pow(2)
		loss = (loss * latent_mask.float()).sum() / latent_mask.float().sum().clamp_min(1.0)

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
