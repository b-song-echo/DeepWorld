import importlib
import sys
import types
from pathlib import Path

import torch


def resolve_torch_dtype(name: str) -> torch.dtype:
	"""Convert a user-facing dtype string into a `torch.dtype`.

	Args:
		name: String such as `bf16`, `bfloat16`, `fp16`, or `float32`.

	Returns:
		The matching `torch.dtype`.

	Raises:
		ValueError: If the string is not recognized.
	"""
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


def ensure_local_vggt_importable(repo_root: str | Path | None = None, local_source_dir: str = "vggt-main") -> None:
	"""Add the local VGGT source tree to `sys.path` if it exists.

	Args:
		repo_root: Optional project root. If omitted, inferred from this file location.
		local_source_dir: Relative directory containing the VGGT source checkout.
	"""
	root = Path(repo_root or Path(__file__).resolve().parents[2])
	source_dir = root / local_source_dir
	if source_dir.exists() and str(source_dir) not in sys.path:
		sys.path.insert(0, str(source_dir))


def _patch_legacy_urllib3_namespace() -> None:
	"""Install a small runtime shim for older `urllib3.packages.*` imports.

	This exists because the current environment has a partially incompatible
	`requests` / `urllib3` stack that some `diffusers` imports still trip over.
	The shim is intentionally minimal and only patches the names needed by the
	current runtime.
	"""
	try:
		import six
	except Exception:
		return

	packages = sys.modules.setdefault("urllib3.packages", types.ModuleType("urllib3.packages"))
	if not hasattr(packages, "__path__"):
		packages.__path__ = []

	six_pkg = sys.modules.setdefault("urllib3.packages.six", types.ModuleType("urllib3.packages.six"))
	if not hasattr(six_pkg, "__path__"):
		six_pkg.__path__ = []

	for attr in ["b", "iterkeys", "itervalues", "PY3", "string_types", "integer_types", "raise_from"]:
		if hasattr(six, attr):
			setattr(six_pkg, attr, getattr(six, attr))
	six_pkg.moves = six.moves

	moves_pkg = sys.modules.setdefault("urllib3.packages.six.moves", types.ModuleType("urllib3.packages.six.moves"))
	if not hasattr(moves_pkg, "__path__"):
		moves_pkg.__path__ = []

	for name in ["http_client", "queue", "urllib", "urllib_parse", "urllib_error", "urllib_robotparser"]:
		target = getattr(six.moves, name, None)
		if target is not None:
			sys.modules[f"urllib3.packages.six.moves.{name}"] = target
			setattr(moves_pkg, name, target)

	try:
		import ssl_match_hostname

		sys.modules["urllib3.packages.ssl_match_hostname"] = ssl_match_hostname
	except Exception:
		pass


def load_diffusers_classes():
	"""Import the Wan diffusers classes used by the prototype.

	The function first attempts a normal import. If that fails, it applies a
	small compatibility shim for the current environment and retries once.

	Returns:
		A tuple of `(AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel)`.

	Raises:
		RuntimeError: If the classes still cannot be imported.
	"""
	try:
		from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel

		return AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel
	except Exception as first_error:
		_patch_legacy_urllib3_namespace()
		try:
			from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel

			return AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel
		except Exception as second_error:
			raise RuntimeError(
				"Failed to import diffusers Wan classes. The current environment appears to have a broken "
				"`requests`/`urllib3` stack. Fix the environment or install a clean diffusers runtime before training."
			) from second_error


def load_transformers_module(module_path: str):
	"""Import a transformers submodule by string path.

	Args:
		module_path: Fully qualified Python module path.

	Returns:
		The imported module object.
	"""
	return importlib.import_module(module_path)
