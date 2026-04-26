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
) -> list[Image.Image]:
	"""Sample reference images from decoded video frames.

	Args:
		raw_frames: Decoded RGB frames from the source video.
		num_reference_images: Maximum number of reference frames to select.
		random_selection: Whether to sample frames randomly instead of uniformly.

	Returns:
		A list of RGB reference images.
	"""

	if len(raw_frames) == 0:
		raise ValueError("Cannot sample reference images from an empty frame list.")

	selection_count = min(num_reference_images, len(raw_frames))
	if random_selection:
		indices = sorted(random.sample(range(len(raw_frames)), k=selection_count))
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

	span = clip_frames * stride
	if num_frames <= span:
		return list(range(0, num_frames, stride))

	max_start = max(num_frames - span, 0)
	start = random.randint(0, max_start) if random_clip else max_start // 2
	return list(range(start, min(start + span, num_frames), stride))


def load_video_frames(
	path: str | Path,
	num_frames: int,
	stride: int,
	height: int,
	width: int,
	random_clip: bool = True,
) -> Tensor:
	"""Load a video clip, sample frames, and convert it into a training tensor.

	Args:
		path: Video file path.
		num_frames: Requested number of output frames before alignment.
		stride: Temporal sampling stride.
		height: Output frame height.
		width: Output frame width.
		random_clip: Whether to sample a random temporal window.

	Returns:
		A tensor with shape `(T, 3, H, W)` normalized to `[-1, 1]`.
	"""

	raw_frames = decode_video_frames(path)
	indices = _sample_clip_indices(len(raw_frames), num_frames, stride, random_clip=random_clip)
	frames = [raw_frames[index] for index in indices]
	frames = align_frame_count(frames)
	frame_tensors = [prepare_image_tensor(frame, height, width, normalize_to_neg_one=True) for frame in frames]
	return torch.stack(frame_tensors, dim=0)


def load_video_frames_from_raw_frames(
	raw_frames: list[Image.Image],
	num_frames: int,
	stride: int,
	height: int,
	width: int,
	random_clip: bool = True,
) -> Tensor:
	"""Convert decoded frames into a training clip tensor.

	Args:
		raw_frames: Decoded RGB frames from the source video.
		num_frames: Requested number of output frames before alignment.
		stride: Temporal sampling stride.
		height: Output frame height.
		width: Output frame width.
		random_clip: Whether to sample a random temporal window.

	Returns:
		A tensor with shape `(T, 3, H, W)` normalized to `[-1, 1]`.
	"""

	if len(raw_frames) == 0:
		raise ValueError("Cannot build a training clip from an empty frame list.")

	indices = _sample_clip_indices(len(raw_frames), num_frames, stride, random_clip=random_clip)
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


def save_video_tensor(video: Tensor, path: str | Path, fps: int = 16) -> None:
	"""Write one generated video tensor to an MP4 file.

	Args:
		video: Tensor with shape `(3, T, H, W)` or `(T, 3, H, W)` in `[-1, 1]`.
		path: Destination `.mp4` path.
		fps: Output video frame rate.
	"""

	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	iio.imwrite(path, video_tensor_to_uint8(video), fps=fps)
