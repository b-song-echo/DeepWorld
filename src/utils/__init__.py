import torch

from src.utils.video import center_crop_and_resize, load_image, load_video_frames


# TODO: There is really no need for this function, it basically does nothing, why do you even do this... Just import them at the top of a file the way you do with transformers, OK? Again, for the entire implementation, get rid of those redundant checks and functions, and keep the code nice and clean, OK? 
def load_diffusers_classes():
	"""Import the Wan diffusers classes used by the prototype.

	Returns:
		A tuple of `(AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel)`.
	"""

	from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel

	return AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel


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
	"load_diffusers_classes",
	"load_image",
	"load_video_frames",
	"resolve_torch_dtype",
]
