import os

import torch

from src.utils.video import center_crop_and_resize, load_image, load_video_frames

# TODO: Migrate the content in this file to utils.py. Update all references in this codebase. Then delete this file.


def get_world_size() -> int:
	"""Return the active distributed world size.

	Returns:
		The distributed world size, or `1` for single-process execution.
	"""

	if torch.distributed.is_available() and torch.distributed.is_initialized():
		return max(int(torch.distributed.get_world_size()), 1)
	return max(int(os.environ.get("WORLD_SIZE", "1")), 1)


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


__all__ = [
	"center_crop_and_resize",
	"get_world_size",
	"load_image",
	"load_video_frames",
	"resolve_torch_dtype",
]
