import os
import random
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


def load_image(path: str | Path) -> Image.Image:
	"""Load an RGB image from disk.

	Args:
		path: Filesystem path to the image.

	Returns:
		A `PIL.Image` converted to RGB.
	"""

	with Image.open(path) as image:
		return image.convert("RGB")


def center_crop_to_aspect(image: Image.Image, height: int, width: int) -> Image.Image:
	"""Center-crop an image to the requested aspect ratio.

	Args:
		image: Source image.
		height: Target height.
		width: Target width.

	Returns:
		A cropped `PIL.Image` with the requested aspect ratio.
	"""

	source_w, source_h = image.size
	target_ratio = width / height
	source_ratio = source_w / source_h

	if source_ratio > target_ratio:
		crop_w = int(source_h * target_ratio)
		crop_h = source_h
	else:
		crop_w = source_w
		crop_h = int(source_w / target_ratio)

	left = max((source_w - crop_w) // 2, 0)
	top = max((source_h - crop_h) // 2, 0)
	return image.crop((left, top, left + crop_w, top + crop_h))


def resize_image(image: Image.Image, height: int, width: int) -> Image.Image:
	"""Resize an image to the requested spatial shape."""

	return image.resize((width, height), resample=Image.BICUBIC)


def center_crop_and_resize(image: Image.Image, height: int, width: int) -> Image.Image:
	"""Center-crop an image to the requested aspect ratio and resize it.

	Args:
		image: Source image.
		height: Target height.
		width: Target width.

	Returns:
		A resized `PIL.Image` with the requested spatial shape.
	"""

	return resize_image(center_crop_to_aspect(image, height, width), height, width)


def pil_to_tensor(image: Image.Image, normalize_to_neg_one: bool = False) -> Tensor:
	"""Convert a PIL image into a CHW float tensor.

	Args:
		image: Source RGB image.
		normalize_to_neg_one: Whether to map pixel values from `[0, 1]` to `[-1, 1]`.

	Returns:
		A tensor with shape `(3, H, W)`.
	"""

	array = np.asarray(image, dtype=np.float32) / 255.0
	tensor = torch.from_numpy(array).permute(2, 0, 1)
	if normalize_to_neg_one:
		tensor = tensor * 2.0 - 1.0
	return tensor


def prepare_image_tensor(
	image: Image.Image,
	height: int,
	width: int,
	normalize_to_neg_one: bool = False,
) -> Tensor:
	"""Crop, resize, and convert an image into a float tensor.

	Args:
		image: Source RGB image.
		height: Target height.
		width: Target width.
		normalize_to_neg_one: Whether to map pixel values from `[0, 1]` to `[-1, 1]`.

	Returns:
		A tensor with shape `(3, height, width)`.
	"""

	image = center_crop_and_resize(image, height, width)
	return pil_to_tensor(image, normalize_to_neg_one=normalize_to_neg_one)


def decode_video_frames(source: str | Path | bytes) -> list[Image.Image]:
	"""Decode all frames from a video source into RGB PIL images.

	Args:
		source: Video path on disk or raw `.mp4` bytes.

	Returns:
		A list of decoded RGB frames.

	Raises:
		ValueError: If the decoded video has no frames.
	"""

	if isinstance(source, (str, Path)):
		raw_frames = [Image.fromarray(frame).convert("RGB") for frame in iio.imiter(source)]
	else:
		temp_path = None
		try:
			with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
				handle.write(source)
				temp_path = handle.name
			raw_frames = [Image.fromarray(frame).convert("RGB") for frame in iio.imiter(temp_path)]
		finally:
			if temp_path is not None and os.path.exists(temp_path):
				os.remove(temp_path)

	if len(raw_frames) == 0:
		raise ValueError("No frames found in video source.")
	return raw_frames


def sample_reference_images(
	raw_frames: list[Image.Image],
	num_reference_images: int,
	random_selection: bool = True,
	preserve_order: bool = True,
) -> list[Image.Image]:
	"""Sample reference images from decoded video frames.

	Args:
		raw_frames: Decoded RGB frames from the source video.
		num_reference_images: Maximum number of reference frames to select.
		random_selection: Whether to sample frames randomly instead of uniformly.
		preserve_order: Whether selected frames should keep their temporal order.

	Returns:
		A list of RGB reference images.
	"""

	if len(raw_frames) == 0:
		raise ValueError("Cannot sample reference images from an empty frame list.")

	selection_count = min(num_reference_images, len(raw_frames))
	if random_selection:
		indices = random.sample(range(len(raw_frames)), k=selection_count)
		if preserve_order:
			indices = sorted(indices)
	else:
		if selection_count == 1:
			indices = [len(raw_frames) // 2]
		else:
			indices = [round(index * (len(raw_frames) - 1) / (selection_count - 1)) for index in range(selection_count)]
	return [raw_frames[index].copy() for index in indices]


def align_frame_count(frames: list[Image.Image], temporal_factor: int = 4) -> list[Image.Image]:
	"""Adjust the frame count so `num_frames - 1` is divisible by the temporal factor.

	This mirrors Wan-style video latent shapes where the first frame is treated
	specially and temporal compression applies to the remaining frames.

	Args:
		frames: Ordered list of video frames.
		temporal_factor: Temporal compression factor used by the VAE.

	Returns:
		A frame list truncated or padded with the last frame to match the rule.
	"""

	if not frames:
		return frames

	target = ((len(frames) - 1) // temporal_factor) * temporal_factor + 1
	target = max(target, 1)
	if target == len(frames):
		return frames
	if target < len(frames):
		return frames[:target]

	while len(frames) < target:
		frames.append(frames[-1].copy())
	return frames


def _sample_clip_indices(num_frames: int, clip_frames: int, stride: int, random_clip: bool) -> list[int]:
	"""Choose frame indices for a fixed-length training clip.

	Args:
		num_frames: Number of frames available in the source video.
		clip_frames: Requested number of output frames before final alignment.
		stride: Temporal stride between sampled frames.
		random_clip: Whether to randomize the clip start.

	Returns:
		A list of source-frame indices.
	"""

	if num_frames <= 0:
		raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
	if clip_frames <= 0:
		raise ValueError(f"`clip_frames` must be positive, got {clip_frames}.")
	if stride <= 0:
		raise ValueError(f"`stride` must be positive, got {stride}.")

	span = clip_frames * stride
	if num_frames <= span:
		return list(range(0, num_frames, stride))

	max_start = max(num_frames - span, 0)
	start = random.randint(0, max_start) if random_clip else max_start // 2
	return list(range(start, min(start + span, num_frames), stride))


def _sample_uniform_indices(num_frames: int, clip_frames: int) -> list[int]:
	"""Choose frame indices evenly across the whole source video.

	Args:
		num_frames: Number of frames available in the source video.
		clip_frames: Requested number of output frames before final alignment.

	Returns:
		A list of source-frame indices covering the full temporal extent.
	"""

	if num_frames <= 0:
		raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
	if clip_frames <= 0:
		raise ValueError(f"`clip_frames` must be positive, got {clip_frames}.")
	if num_frames <= clip_frames:
		return list(range(num_frames))
	if clip_frames == 1:
		return [num_frames // 2]
	return [round(index * (num_frames - 1) / (clip_frames - 1)) for index in range(clip_frames)]


def sample_video_frame_indices(
	num_source_frames: int,
	num_output_frames: int,
	stride: int | None = None,
	random_clip: bool | None = None,
	frame_sampling: str = "clip",
) -> list[int]:
	"""Choose source-video frame indices according to the requested strategy.

	Args:
		num_source_frames: Number of decoded frames available in the source video.
		num_output_frames: Requested number of output frames before final alignment.
		stride: Optional temporal stride used by the `clip` strategy. Defaults to `1`.
		random_clip: Optional `clip` strategy flag controlling whether to randomize the window start. Defaults to `True`.
		frame_sampling: Sampling strategy, either `clip` or `uniform`.

	Returns:
		A list of source-frame indices.

	Raises:
		ValueError: If `frame_sampling` is unsupported.
	"""

	if frame_sampling == "clip":
		clip_stride = 1 if stride is None else stride
		clip_random = True if random_clip is None else random_clip
		return _sample_clip_indices(
			num_source_frames,
			num_output_frames,
			clip_stride,
			random_clip=clip_random,
		)
	if frame_sampling == "uniform":
		return _sample_uniform_indices(num_source_frames, num_output_frames)
	raise ValueError(f"Unsupported frame sampling strategy: {frame_sampling!r}.")


def load_video_frames(
	path: str | Path,
	num_frames: int,
	height: int,
	width: int,
	*,
	stride: int | None = None,
	random_clip: bool | None = None,
	frame_sampling: str = "clip",
) -> Tensor:
	"""Load a video clip, sample frames, and convert it into a training tensor.

	Args:
		path: Video file path.
		num_frames: Requested number of output frames before alignment.
		height: Output frame height.
		width: Output frame width.
		stride: Optional temporal stride used by the `clip` strategy. Defaults to `1`.
		random_clip: Optional `clip` strategy flag controlling whether to randomize the window start. Defaults to `True`.
		frame_sampling: Sampling strategy, either `clip` or `uniform`.

	Returns:
		A tensor with shape `(T, 3, H, W)` normalized to `[-1, 1]`.
	"""

	raw_frames = decode_video_frames(path)
	indices = sample_video_frame_indices(
		len(raw_frames),
		num_frames,
		stride=stride,
		random_clip=random_clip,
		frame_sampling=frame_sampling,
	)
	frames = [raw_frames[index] for index in indices]
	frames = align_frame_count(frames)
	frame_tensors = [prepare_image_tensor(frame, height, width, normalize_to_neg_one=True) for frame in frames]
	return torch.stack(frame_tensors, dim=0)


def load_video_frames_from_raw_frames(
	raw_frames: list[Image.Image],
	num_frames: int,
	height: int,
	width: int,
	*,
	stride: int | None = None,
	random_clip: bool | None = None,
	frame_sampling: str = "clip",
) -> Tensor:
	"""Convert decoded frames into a training clip tensor.

	Args:
		raw_frames: Decoded RGB frames from the source video.
		num_frames: Requested number of output frames before alignment.
		height: Output frame height.
		width: Output frame width.
		stride: Optional temporal stride used by the `clip` strategy. Defaults to `1`.
		random_clip: Optional `clip` strategy flag controlling whether to randomize the window start. Defaults to `True`.
		frame_sampling: Sampling strategy, either `clip` or `uniform`.

	Returns:
		A tensor with shape `(T, 3, H, W)` normalized to `[-1, 1]`.
	"""

	if len(raw_frames) == 0:
		raise ValueError("Cannot build a training clip from an empty frame list.")

	indices = sample_video_frame_indices(
		len(raw_frames),
		num_frames,
		stride=stride,
		random_clip=random_clip,
		frame_sampling=frame_sampling,
	)
	frames = [raw_frames[index] for index in indices]
	frames = align_frame_count(frames)
	frame_tensors = [prepare_image_tensor(frame, height, width, normalize_to_neg_one=True) for frame in frames]
	return torch.stack(frame_tensors, dim=0)


def resize_tensor_image(image: Tensor, height: int, width: int) -> Tensor:
	"""Resize a CHW tensor image with bicubic interpolation.

	Args:
		image: Tensor with shape `(C, H, W)`.
		height: Target height.
		width: Target width.

	Returns:
		Resized image tensor with shape `(C, height, width)`.
	"""

	return F.interpolate(image.unsqueeze(0), size=(height, width), mode="bicubic", align_corners=False).squeeze(0)


def video_tensor_to_uint8(video: Tensor) -> np.ndarray:
	"""Convert one generated video tensor into uint8 frames for serialization.

	Args:
		video: Tensor with shape `(3, T, H, W)` or `(T, 3, H, W)` in `[-1, 1]`.

	Returns:
		A NumPy array with shape `(T, H, W, 3)` and dtype `uint8`.
	"""

	if video.ndim != 4:
		raise ValueError(f"Expected a 4D video tensor, got shape {tuple(video.shape)}.")

	video = video.detach().float().cpu().clamp(-1.0, 1.0)
	if video.size(0) == 3:
		frames = video.permute(1, 2, 3, 0)
	elif video.size(1) == 3:
		frames = video.permute(0, 2, 3, 1)
	else:
		raise ValueError(f"Expected RGB channels in dimension 0 or 1, got shape {tuple(video.shape)}.")

	frames = ((frames + 1.0) * 127.5).round().to(torch.uint8)
	return frames.contiguous().numpy()


def image_tensor_to_uint8(image: Tensor) -> np.ndarray:
	"""Convert one image tensor into an RGB uint8 array.

	Args:
		image: Tensor with shape `(3, H, W)` in either `[0, 1]` or `[-1, 1]`.

	Returns:
		A NumPy array with shape `(H, W, 3)` and dtype `uint8`.
	"""

	if image.ndim != 3 or image.size(0) != 3:
		raise ValueError(f"Expected a CHW RGB image tensor, got shape {tuple(image.shape)}.")

	image = image.detach().float().cpu()
	if image.min().item() < 0.0:
		image = image.clamp(-1.0, 1.0).add(1.0).mul(0.5)
	else:
		image = image.clamp(0.0, 1.0)
	image = image.permute(1, 2, 0).mul(255.0).round().to(torch.uint8)
	return image.contiguous().numpy()


def save_image_tensor(image: Tensor, path: str | Path) -> None:
	"""Write one RGB image tensor to disk.

	Args:
		image: Tensor with shape `(3, H, W)` in either `[0, 1]` or `[-1, 1]`.
		path: Destination image path.
	"""

	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	iio.imwrite(path, image_tensor_to_uint8(image))


def save_video_tensor(
	video: Tensor,
	path: str | Path,
	fps: float | None = None,
	duration_seconds: float | None = None,
) -> None:
	"""Write one generated video tensor to an MP4 file.

	Args:
		video: Tensor with shape `(3, T, H, W)` or `(T, 3, H, W)` in `[-1, 1]`.
		path: Destination `.mp4` path.
		fps: Optional output frame rate. Defaults to `16` when `duration_seconds` is omitted.
		duration_seconds: Optional target playback duration used to derive FPS from the frame count.
	"""

	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	frames = video_tensor_to_uint8(video)
	
	if fps is not None and duration_seconds is not None:
		raise ValueError("Only one of `fps` or `duration_seconds` may be provided.")
	if duration_seconds is not None:
		if duration_seconds <= 0:
			raise ValueError(f"`duration_seconds` must be positive, got {duration_seconds}.")
		fps = frames.shape[0] / duration_seconds
	if fps is None:
		fps = 16
	if fps <= 0:
		raise ValueError(f"`fps` must be positive, got {fps}.")
	iio.imwrite(path, frames, fps=fps)
