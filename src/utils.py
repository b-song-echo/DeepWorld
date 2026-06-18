import json
import os
import random
import tempfile
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TextIO

import imageio.v3 as iio
import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor


def get_world_size() -> int:
	"""Return the active distributed world size.

	Returns:
		The distributed world size, or `1` for single-process execution.
	"""

	if torch.distributed.is_available() and torch.distributed.is_initialized():
		return max(int(torch.distributed.get_world_size()), 1)
	return max(int(os.environ.get("WORLD_SIZE", "1")), 1)


def configure_offline_runtime() -> None:
	"""Apply process-wide defaults for offline, CUDA-oriented training runs."""

	# TODO: Since these are set here, there is no need to set them repeatedly in the bash script. Check the entire codebase for redundant code and remove them. For example, if the configuration values are checked at data class post initialization, there is no need to recheck them again and again when using them. This can also be used in `synthesize.py`, then these environment variables do not need to be set in bash scripts.
	os.environ.setdefault("HF_HUB_OFFLINE", "1")
	os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
	os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
	os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

	warnings.filterwarnings("ignore")
	try:
		from diffusers.utils import logging
		logging.set_verbosity_error()
	except Exception:
		pass

	torch.multiprocessing.set_sharing_strategy("file_system")
	if torch.cuda.is_available():
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = False


def seed_python_and_torch(seed: int) -> None:
	"""Seed Python and torch RNGs for a single process."""

	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def save_config_yaml(config: Any, output_dir: str | Path, filename: str = "configs.yaml") -> None:
	"""Write a dataclass or mapping config snapshot into a run directory."""

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	payload = asdict(config) if is_dataclass(config) else config
	with (output_dir / filename).open("w", encoding="utf-8") as handle:
		yaml.safe_dump(payload, handle, sort_keys=False)


def open_rank0_jsonl_log(output_dir: str | Path, is_main_process: bool) -> TextIO | None:
	"""Open the rank-0 JSONL metrics log, returning `None` on non-main ranks."""

	if not is_main_process:
		return None
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	return (output_dir / "logs.jsonl").open("a", encoding="utf-8", buffering=1)


def write_jsonl_log(handle: TextIO | None, payload: dict[str, Any]) -> None:
	"""Append one JSONL metrics event and force it to disk."""

	if handle is None:
		return
	handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
	handle.flush()
	os.fsync(handle.fileno())


def read_json_file(path: str | Path) -> Any:
	"""Read one JSON file."""

	with Path(path).open("r", encoding="utf-8") as handle:
		return json.load(handle)


def write_json_file(path: str | Path, payload: Any) -> None:
	"""Write one formatted JSON file."""

	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, ensure_ascii=False)
		handle.write("\n")


def read_jsonl_file(path: str | Path, require_objects: bool = True) -> list[dict[str, Any]]:
	"""Read one JSONL file into entry dictionaries."""

	path = Path(path)
	if not path.exists():
		return []
	entries: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line_index, line in enumerate(handle, start=1):
			if not line.strip():
				continue
			entry = json.loads(line)
			if require_objects and not isinstance(entry, dict):
				raise ValueError(f"JSONL entry {path}:{line_index} is not a JSON object.")
			entries.append(entry)
	return entries


def write_jsonl_file(path: str | Path, *payload: dict[str, Any], append: bool = True) -> None:
	"""Write or append JSONL entries and force them to disk."""

	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a" if append else "w", encoding="utf-8", buffering=1) as handle:
		for entry in payload:
			handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
		handle.flush()
		os.fsync(handle.fileno())


def resolve_torch_dtype(name: str | None) -> torch.dtype | None:
	"""Convert a config dtype string into a `torch.dtype`.

	Args:
		name: String such as `bf16`, `bfloat16`, `fp16`, or `float32`, or `None`.

	Returns:
		The matching `torch.dtype`, or `None` when the checkpoint default should be used.
	"""

	if name is None:
		return None

	table = {
		"float16": torch.float16,
		"fp16": torch.float16,
		"bfloat16": torch.bfloat16,
		"bf16": torch.bfloat16,
		"float32": torch.float32,
		"fp32": torch.float32,
	}
	key = name.lower()
	if key not in table:
		raise ValueError(f"Unsupported torch dtype: {name}")
	return table[key]


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
	"""Resize an image to the requested spatial shape.

	Args:
		image: Source image.
		height: Target height.
		width: Target width.

	Returns:
		The resized RGB image.
	"""

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
	"""

	def sample_clip_indices(num_frames: int, clip_frames: int, clip_stride: int, use_random_clip: bool) -> list[int]:
		"""Choose frame indices for one fixed-length training clip."""

		span = clip_frames * clip_stride
		if num_frames <= span:
			return list(range(0, num_frames, clip_stride))

		max_start = max(num_frames - span, 0)
		start = random.randint(0, max_start) if use_random_clip else max_start // 2
		return list(range(start, min(start + span, num_frames), clip_stride))

	def sample_uniform_indices(num_frames: int, clip_frames: int) -> list[int]:
		"""Choose frame indices evenly across the source video."""

		if num_frames <= clip_frames:
			return list(range(num_frames))
		if clip_frames == 1:
			return [num_frames // 2]
		return [round(index * (num_frames - 1) / (clip_frames - 1)) for index in range(clip_frames)]

	if num_source_frames <= 0:
		raise ValueError(f"`num_source_frames` must be positive, got {num_source_frames}.")
	if num_output_frames <= 0:
		raise ValueError(f"`num_output_frames` must be positive, got {num_output_frames}.")

	if frame_sampling == "clip":
		clip_stride = 1 if stride is None else stride
		if clip_stride <= 0:
			raise ValueError(f"`stride` must be positive, got {clip_stride}.")
		clip_random = True if random_clip is None else random_clip
		return sample_clip_indices(num_source_frames, num_output_frames, clip_stride, clip_random)
	if frame_sampling == "uniform":
		return sample_uniform_indices(num_source_frames, num_output_frames)
	raise ValueError(f"Unsupported frame sampling strategy: {frame_sampling!r}.")


def align_frame_count(frames: list[Image.Image], temporal_factor: int = 4) -> list[Image.Image]:
	"""Adjust frame count so `num_frames - 1` is divisible by a temporal factor.

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


def sample_video_frames_from_raw_frames(
	raw_frames: list[Image.Image],
	num_frames: int,
	*,
	stride: int | None = None,
	random_clip: bool | None = None,
	frame_sampling: str = "clip",
) -> list[Image.Image]:
	"""Select and temporally align frames from an already decoded video.

	Args:
		raw_frames: Decoded RGB frames from the source video.
		num_frames: Requested number of output frames before Wan temporal alignment.
		stride: Optional temporal stride used by the `clip` strategy. Defaults to `1`.
		random_clip: Optional `clip` strategy flag controlling whether to randomize the window start. Defaults to `True`.
		frame_sampling: Sampling strategy, either `clip` or `uniform`.

	Returns:
		An ordered frame list after Wan-compatible temporal alignment.
	"""

	if len(raw_frames) == 0:
		raise ValueError("Cannot sample frames from an empty frame list.")

	indices = sample_video_frame_indices(
		len(raw_frames),
		num_frames,
		stride=stride,
		random_clip=random_clip,
		frame_sampling=frame_sampling,
	)
	frames = [raw_frames[index] for index in indices]
	return align_frame_count(frames)


def video_frames_to_tensor(frames: list[Image.Image], height: int, width: int) -> Tensor:
	"""Convert an ordered RGB frame list into the normalized training tensor.

	Args:
		frames: Selected RGB frames.
		height: Output frame height.
		width: Output frame width.

	Returns:
		A tensor with shape `(T, 3, H, W)` normalized to `[-1, 1]`.
	"""

	if len(frames) == 0:
		raise ValueError("Cannot build a video tensor from an empty frame list.")

	frame_tensors = [prepare_image_tensor(frame, height, width, normalize_to_neg_one=True) for frame in frames]
	return torch.stack(frame_tensors, dim=0)


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

	frames = sample_video_frames_from_raw_frames(
		decode_video_frames(path),
		num_frames,
		stride=stride,
		random_clip=random_clip,
		frame_sampling=frame_sampling,
	)
	return video_frames_to_tensor(frames, height, width)


def video_tensor_to_uint8(video: Tensor) -> np.ndarray:
	"""Convert one generated video tensor into uint8 frames for serialization.

	Args:
		video: Tensor with shape `(3, T, H, W)` or `(T, 3, H, W)` in `[-1, 1]`.

	Returns:
		A NumPy array with shape `(T, H, W, 3)` and dtype `uint8`.
	"""

	if video.dim() != 4:
		raise ValueError(f"Expected a 4D video tensor, got shape {tuple(video.size())}.")

	video = video.detach().float().cpu().clamp(-1.0, 1.0)
	if video.size(0) == 3:
		frames = video.permute(1, 2, 3, 0)
	elif video.size(1) == 3:
		frames = video.permute(0, 2, 3, 1)
	else:
		raise ValueError(f"Expected RGB channels in dimension 0 or 1, got shape {tuple(video.size())}.")

	frames = ((frames + 1.0) * 127.5).round().to(torch.uint8)
	return frames.contiguous().numpy()


def image_tensor_to_uint8(image: Tensor) -> np.ndarray:
	"""Convert one image tensor into an RGB uint8 array.

	Args:
		image: Tensor with shape `(3, H, W)` in either `[0, 1]` or `[-1, 1]`.

	Returns:
		A NumPy array with shape `(H, W, 3)` and dtype `uint8`.
	"""

	if image.dim() != 3 or image.size(0) != 3:
		raise ValueError(f"Expected a CHW RGB image tensor, got shape {tuple(image.size())}.")

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
		fps = len(frames) / duration_seconds
	if fps is None:
		fps = 16
	if fps <= 0:
		raise ValueError(f"`fps` must be positive, got {fps}.")
	iio.imwrite(path, frames, fps=fps)
