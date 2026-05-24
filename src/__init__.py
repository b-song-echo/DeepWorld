from src.config import WorldModelConfig, load_config


def __getattr__(name: str):
	"""Lazily import heavy model classes only when requested."""

	if name == "DeepWorld":
		from src.models.deep_world import DeepWorld
		return DeepWorld
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["DeepWorld", "WorldModelConfig", "load_config"]
