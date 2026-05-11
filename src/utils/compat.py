import os
import importlib
import sys
import types
from pathlib import Path

import torch


# TODO: Put this function to utils/__init__.py because it is not a compatability related function
def resolve_torch_dtype(name: str | None) -> torch.dtype | None:
	"""Convert a user-facing dtype string into a `torch.dtype`.

	Args:
		name: String such as `bf16`, `bfloat16`, `fp16`, or `float32`, or `None`.

	Returns:
		The matching `torch.dtype`, or `None` when the checkpoint default should be used.

	Raises:
		ValueError: If the string is not recognized.
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


# TODO: Put this function directly in train.py
def get_world_size() -> int:
	"""Return the active distributed world size.

	The helper prefers the initialized torch.distributed process group when
	available, and otherwise falls back to the launcher-provided `WORLD_SIZE`
	environment variable.

	Returns:
		The distributed world size, or `1` for single-process execution.
	"""

	if torch.distributed.is_available() and torch.distributed.is_initialized():
		return max(int(torch.distributed.get_world_size()), 1)
	return max(int(os.environ.get("WORLD_SIZE", "1")), 1)


# TODO: There is no need for this check. VGGT directory is always there is directly importable. Get rid of these function.
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


# TODO: Just get rid of this check, I won't run this program on this local Mac, instead, I will run it on a linux server with CUDA support.
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

	urllib_pkg = getattr(six.moves, "urllib", None)
	if urllib_pkg is not None:
		sys.modules["urllib3.packages.six.moves.urllib"] = urllib_pkg
		setattr(moves_pkg, "urllib", urllib_pkg)
		for submodule_name in ["parse", "error", "request", "robotparser"]:
			target = getattr(urllib_pkg, submodule_name, None)
			if target is None:
				target = getattr(six.moves, f"urllib_{submodule_name}", None)
			if target is None:
				continue
			sys.modules[f"urllib3.packages.six.moves.urllib.{submodule_name}"] = target

	try:
		import ssl_match_hostname

		sys.modules["urllib3.packages.ssl_match_hostname"] = ssl_match_hostname
	except Exception:
		pass


# TODO: Just get rid of this check, I won't run this program on this local Mac, instead, I will run it on a linux server with CUDA support.
def _patch_transformers_hybrid_cache() -> None:
	"""Expose a minimal `HybridCache` symbol for PEFT/diffusers version skew.

	The local macOS environment can pair a PEFT build that imports
	`transformers.HybridCache` with a Transformers build that no longer exports
	that symbol. DeepWorld does not use PEFT prompt-tuning caches, but diffusers
	imports PEFT mixins while loading Wan classes. Providing the symbol keeps that
	import path usable without affecting the project's own cache behavior.
	"""

	try:
		import transformers
	except Exception:
		return

	if hasattr(transformers, "HybridCache") or not hasattr(transformers, "DynamicCache"):
		return

	class HybridCache(transformers.DynamicCache):
		"""Compatibility cache shim used only when PEFT imports require the name."""

		def __init__(
			self,
			config=None,
			max_batch_size=None,
			max_cache_len=None,
			dtype=None,
			device=None,
			**kwargs,
		):
			super().__init__(config=config)
			self.max_batch_size = max_batch_size
			self.max_cache_len = max_cache_len
			self.dtype = dtype
			self.device = device

	transformers.HybridCache = HybridCache


# TODO: This is unnecessary, I'll run this program on a CUDA environment with diffusers or transformers properly installed and no compatability issues. Remove this check.
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
		_patch_transformers_hybrid_cache()
		try:
			from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel

			return AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel
		except Exception as second_error:
			raise RuntimeError(
				"Failed to import diffusers Wan classes after applying local compatibility shims. "
				"Fix the Hugging Face package versions or install a clean diffusers runtime before training."
			) from second_error


# TODO: there is no need for this safety check. I won't run it on this Mac, so the errors you see are not real issues. Just ignore them.
def load_transformers_module(module_path: str):
	"""Import a transformers submodule by string path.

	Args:
		module_path: Fully qualified Python module path.

	Returns:
		The imported module object.
	"""
	return importlib.import_module(module_path)
