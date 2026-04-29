import json
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt


LOG_PATH = Path("/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/exps/demo/logs.jsonl")
METRIC_NAMES = ("loss",)
FIGURE_SIZE = (10, 4)


def include_entry(entry: dict) -> bool:
	"""Return whether a JSONL log entry should be plotted."""

	return entry.get("event") == "train"


def load_entries(path: Path) -> list[dict]:
	"""Load JSONL training-log entries from disk.

	Args:
		path: Path to the `logs.jsonl` file written by `train.py`.

	Returns:
		A list of decoded log entries.
	"""

	entries: list[dict] = []
	with path.open("r", encoding="utf-8") as handle:
		for line_number, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				entries.append(json.loads(line))
			except json.JSONDecodeError as error:
				raise ValueError(f"Invalid JSON on line {line_number} of {path}.") from error
	return entries


def metric_series(
	entries: Iterable[dict],
	metric_name: str,
	entry_filter: Callable[[dict], bool],
) -> tuple[list[int], list[float]]:
	"""Extract one metric's `(step, value)` series from log entries.

	Args:
		entries: Decoded JSONL entries.
		metric_name: Metric key to plot.
		entry_filter: Predicate used to select entries.

	Returns:
		Two aligned lists containing steps and metric values.
	"""

	steps: list[int] = []
	values: list[float] = []
	for entry in entries:
		if not entry_filter(entry):
			continue
		if "step" not in entry or metric_name not in entry:
			continue
		steps.append(int(entry["step"]))
		values.append(float(entry[metric_name]))
	return steps, values


def main() -> None:
	"""Plot hardcoded metrics from the hardcoded JSONL log path."""

	# TODO: each metric is a standalone figure and is shown separately, figsize applies to each individual figrue.
	if not LOG_PATH.exists():
		raise FileNotFoundError(f"Log file does not exist: {LOG_PATH}")

	entries = load_entries(LOG_PATH)
	fig, axes = plt.subplots(len(METRIC_NAMES), 1, figsize=FIGURE_SIZE, squeeze=False)
	for axis, metric_name in zip(axes[:, 0], METRIC_NAMES):
		steps, values = metric_series(entries, metric_name, include_entry)
		if len(steps) == 0:
			axis.set_title(f"{metric_name}: no matching entries")
			continue
		axis.plot(steps, values, linewidth=1.5)
		axis.set_title(metric_name)
		axis.set_xlabel("step")
		axis.set_ylabel(metric_name)
		axis.grid(True, alpha=0.3)

	fig.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
