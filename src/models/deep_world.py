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

	if config.wan.train_scheduler_steps is not None:
		return int(config.wan.train_scheduler_steps)

	load_kwargs = {
		"subfolder": "scheduler",
		"local_files_only": True,
	}
	try:
		if hasattr(scheduler_cls, "from_pretrained"):
			scheduler = scheduler_cls.from_pretrained(config.wan.checkpoint_path, **load_kwargs)
			return int(scheduler.config.num_train_timesteps)
		if hasattr(scheduler_cls, "load_config"):
			scheduler_config = scheduler_cls.load_config(config.wan.checkpoint_path, **load_kwargs)
			return int(scheduler_config["num_train_timesteps"])
	except Exception as error:
		import warnings
		warnings.warn(
			f"Failed to infer Wan scheduler timesteps from {config.wan.checkpoint_path!r}: {error}. "
			"Falling back to 1000 training steps.",
			RuntimeWarning,
		)

	return 1000


class DeepWorld(nn.Module):
	"""Top-level grounded video model combining Qwen, VGGT, Wan, and the Wan VAE.

	Args:
		config: Root configuration object describing all model and training settings.
	"""

	def __init__(self, config: WorldModelConfig):
		super().__init__()
		self.config = config
		self.brain = QwenBrain(
			config.qwen,
			config.vggt,
			gradient_checkpointing=config.training.gradient_checkpointing,
		)

		AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, _ = load_diffusers_classes()
		self.vae = AutoencoderKLWan.from_pretrained(
			config.wan.checkpoint_path,
			subfolder="vae",
			local_files_only=True,
		)
		if config.wan.vae_enable_slicing and hasattr(self.vae, "enable_slicing"):
			self.vae.enable_slicing()
		if config.wan.vae_enable_tiling and hasattr(self.vae, "enable_tiling"):
			self.vae.enable_tiling()
		self.vae.requires_grad_(False)
		train_scheduler_steps = resolve_train_scheduler_steps(config, FlowMatchEulerDiscreteScheduler)
		self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=train_scheduler_steps)

		self.renderer = WanRenderer(
			config.wan,
			condition_dim=self.brain.hidden_size,
			gradient_checkpointing=config.training.gradient_checkpointing,
		)

		latents_mean = torch.tensor(self.vae.config.latents_mean, dtype=torch.float32).view(1, self.vae.config.z_dim, 1, 1, 1)
		latents_recip_std = 1.0 / torch.tensor(self.vae.config.latents_std, dtype=torch.float32).view(
			1, self.vae.config.z_dim, 1, 1, 1
		)

		# NOTE: do not use self.register_buffer, use nn.Buffer instead
		self.latents_mean = nn.Buffer(latents_mean, persistent=False)
		self.latents_recip_std = nn.Buffer(latents_recip_std, persistent=False)

	def encode_videos(self, videos: Tensor, sample_posterior: bool = True) -> Tensor:
		"""Encode GT videos into normalized Wan latent space.

		Args:
			videos: Input video tensor with shape `(B, 3, T, H, W)` in Wan pixel space.
			sample_posterior: Whether to sample from the VAE posterior instead of using its mode.

		Returns:
			Normalized latent tensor with Wan-compatible channel layout.
		"""

		posterior = self.vae.encode(videos.to(dtype=self.vae.dtype, non_blocking=True)).latent_dist
		latents = posterior.sample() if sample_posterior else posterior.mode()
		latents = (latents.float() - self.latents_mean) * self.latents_recip_std
		return latents.to(self.renderer.transformer.patch_embedding.weight.dtype)

	def decode_latents(self, latents: Tensor) -> Tensor:
		"""Decode normalized Wan latents back into videos.

		Args:
			latents: Normalized latent tensor.

		Returns:
			Decoded video tensor in Wan output pixel range.
		"""

		latents = latents / self.latents_recip_std + self.latents_mean
		return self.vae.decode(latents.to(dtype=self.vae.dtype), return_dict=False)[0]

	def _latent_patch_grids(self, latents: Tensor) -> Tensor:
		"""Compute the Wan patch-token grid for a latent tensor.

		Args:
			latents: Latent tensor with shape `(B, C, T, H, W)`.

		Returns:
			A tensor of shape `(B, 3)` containing `(t, h, w)` patch-grid sizes.
		"""

		p_t, p_h, p_w = self.renderer.transformer.config.patch_size
		grid = torch.tensor(
			[
				latents.shape[2] // p_t,
				latents.shape[3] // p_h,
				latents.shape[4] // p_w,
			],
			device=latents.device,
			dtype=torch.long,
		)
		return grid.unsqueeze(0).expand(latents.shape[0], -1)

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
		batch_size, _, latent_frames, latent_height, latent_width = latents.shape
		valid_latent_frames = ((frame_counts.to(device, non_blocking=True) - 1) // self.vae.config.scale_factor_temporal) + 1

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
		latents = self.encode_videos(videos, sample_posterior=True)
		latent_mask, token_mask = self._build_loss_masks(batch["video_frame_counts"], latents)
		latent_patch_grids = self._latent_patch_grids(latents)

		brain_outputs = self.brain(batch, latent_patch_grids=latent_patch_grids)

		noise = torch.randn_like(latents)
		sigmas = torch.rand(latents.shape[0], device=latents.device, dtype=latents.dtype)
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
		num_frames = (num_frames - 1) // self.vae.config.scale_factor_temporal * self.vae.config.scale_factor_temporal + 1

		latent_frames = (num_frames - 1) // self.vae.config.scale_factor_temporal + 1
		latents = torch.randn(
			batch["txt_input_ids"].shape[0],
			self.renderer.transformer.config.in_channels,
			latent_frames,
			height // self.vae.config.scale_factor_spatial,
			width // self.vae.config.scale_factor_spatial,
			device=device,
			dtype=self.renderer.transformer.patch_embedding.weight.dtype,
			generator=generator,
		)

		latent_patch_grids = self._latent_patch_grids(latents)
		brain_outputs = self.brain(batch, latent_patch_grids=latent_patch_grids)
		token_mask = brain_outputs["gen_mask"][:, : brain_outputs["gen_hidden_states"].shape[1]]
		self.scheduler.set_timesteps(num_inference_steps or self.config.wan.inference_steps, device=device)

		for timestep in self.scheduler.timesteps:
			model_output = self.renderer(
				hidden_states=latents,
				timestep=timestep.expand(latents.shape[0]),
				condition_hidden_states=brain_outputs["gen_hidden_states"],
				token_mask=token_mask,
				return_dict=True,
			)["sample"]
			latents = self.scheduler.step(model_output, timestep, latents, return_dict=False)[0]

		return self.decode_latents(latents)
