from src.config import (
	DeepWorldHYConfig,
	DeepWorldQWConfig,
	load_hy_config,
	load_qw_config,
)


def __getattr__(name: str):
	"""Lazily import heavy model classes only when requested."""

	if name == "DeepWorldQW":
		from src.deep_world_qw import DeepWorldQW
		return DeepWorldQW
	if name == "DeepWorldHY":
		from src.deep_world_hy import DeepWorldHY
		return DeepWorldHY
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
	"DeepWorldQW",
	"DeepWorldHY",
	"DeepWorldQWConfig",
	"DeepWorldHYConfig",
	"load_qw_config",
	"load_hy_config",
]
