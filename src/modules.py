from collections.abc import Iterable

import torch.nn as nn
from torch import Tensor


class LoraLinear(nn.Module):
	"""Wrap a frozen linear layer with a trainable low-rank residual branch.

	Args:
		base_layer: Frozen base `nn.Linear` layer.
		rank: LoRA rank.
		alpha: LoRA scaling numerator.
		dropout: Dropout probability applied before the low-rank up projection.
	"""

	def __init__(self, base_layer: nn.Linear, rank: int, alpha: int, dropout: float):
		"""Initialize LoRA matrices beside the frozen base layer."""

		super().__init__()
		self.base_layer = base_layer
		self.base_layer.requires_grad_(False)

		self.rank = rank
		self.scaling = alpha / rank
		self.dropout = nn.Dropout(dropout)
		device = base_layer.weight.device
		dtype = base_layer.weight.dtype

		self.lora_a = nn.Linear(
			base_layer.in_features, rank, bias=False,
			device=device, dtype=dtype,
		)
		self.lora_b = nn.Linear(
			rank, base_layer.out_features, bias=False,
			device=device, dtype=dtype,
		)

		nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
		nn.init.zeros_(self.lora_b.weight)

	def forward(self, hidden_states: Tensor) -> Tensor:
		"""Apply the frozen linear layer plus the trainable LoRA residual."""

		middle = self.dropout(self.lora_a(hidden_states))
		residual = self.lora_b(middle) * self.scaling
		return self.base_layer(hidden_states) + residual


def inject_lora_layers(
	module: nn.Module,
	target_names: Iterable[str],
	rank: int,
	alpha: int,
	dropout: float,
) -> None:
	"""Recursively replace selected linear submodules with `LoraLinear`.

	Args:
		module: Root module to traverse.
		target_names: Child-module names that should receive LoRA wrappers.
		rank: LoRA rank.
		alpha: LoRA scaling numerator.
		dropout: LoRA dropout probability.
	"""

	target_names = set(target_names)
	for name, child in list(module.named_children()):
		if isinstance(child, nn.Linear) and name in target_names:
			lora_child = LoraLinear(child, rank=rank, alpha=alpha, dropout=dropout)
			setattr(module, name, lora_child)
		else:
			inject_lora_layers(child, target_names, rank, alpha, dropout)
