import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import webdataset as wds
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from src.config import DatasetConfig
from src.utils import get_world_size
from src.utils.video import (
	center_crop_to_aspect,
	decode_video_frames,
	load_image,
	load_video_frames,
	pil_to_tensor,
	resize_image,
	sample_reference_images,
	sample_video_frames_from_raw_frames,
	video_frames_to_tensor,
)


def partition_count(total: int, index: int, partitions: int) -> int:
	"""Return the item count assigned to one deterministic partition.

	The first `total % partitions` partitions receive one extra item. This keeps
	the global sample cap exact while preserving a stable assignment for a fixed
	world-size and dataloader-worker configuration.

	Args:
		total: Total number of items to divide.
		index: Zero-based partition index.
		partitions: Number of partitions.

	Returns:
		The item count assigned to `index`.
	"""

	if total <= 0:
		return 0
	if partitions <= 0:
		raise ValueError(f"`partitions` must be positive, got {partitions}.")
	if index < 0 or index >= partitions:
		raise ValueError(f"`index` must be in [0, {partitions}), got {index}.")

	base = total // partitions
	remainder = total % partitions
	return base + int(index < remainder)


def resolve_geo_image_size(dataset_config: DatasetConfig, patch_size: int) -> int:
	"""Resolve the square VGGT input size from config and patch-grid constraints.

	Args:
		dataset_config: Dataset section of the root config.
		patch_size: VGGT patch size that the image resolution must be divisible by.

	Returns:
		A valid VGGT image size divisible by `patch_size`.

	Raises:
		ValueError: If the configured geometry image size is invalid.
	"""

	if patch_size <= 0:
		raise ValueError(f"`patch_size` must be positive, got {patch_size}.")

	if dataset_config.geo_image_size is not None:
		if dataset_config.geo_image_size < patch_size:
			raise ValueError(
				f"`dataset.geo_image_size={dataset_config.geo_image_size}` must be at least the VGGT patch size {patch_size}."
			)
		if dataset_config.geo_image_size % patch_size != 0:
			raise ValueError(
				f"`dataset.geo_image_size={dataset_config.geo_image_size}` must be divisible by the VGGT patch size {patch_size}."
			)
		return dataset_config.geo_image_size

	aligned_size = (dataset_config.vis_image_size // patch_size) * patch_size
	if aligned_size < patch_size:
		raise ValueError(
			f"`dataset.vis_image_size={dataset_config.vis_image_size}` is too small for the VGGT patch size {patch_size}."
		)
	return aligned_size


@dataclass
class WorldModelSample:
	"""In-memory representation of one training sample.

	Attributes:
		sample_id: Stable sample identifier.
		prompt: Text prompt describing scene and camera motion.
		reference_images: List of RGB reference images as PIL objects.
		video: GT video clip tensor with shape `(T, 3, H, W)`.
	"""

	sample_id: str
	prompt: str
	reference_images: list[Any]
	video: Tensor


class ManifestWorldModelDataset(Dataset):
	"""Manifest-driven dataset for grounded video training.

	Each row in the JSONL manifest is expected to provide prompt text, several
	reference image paths, and one ground-truth video path.
	"""

	def __init__(self, config: DatasetConfig):
		"""Initialize the dataset and load the manifest.

		Args:
			config: Dataset section of the root config.
		"""

		self.config = config
		self.records = self._load_manifest(config.manifest_path)
		if config.num_samples > 0:
			self.records = self.records[: config.num_samples]

	def _load_manifest(self, manifest_path: str) -> list[dict[str, Any]]:
		"""Read the JSONL manifest into memory.

		Args:
			manifest_path: Path to the JSONL manifest file.

		Returns:
			A list of parsed records.

		Raises:
			ValueError: If the manifest is empty.
		"""

		records: list[dict[str, Any]] = []
		with Path(manifest_path).open("r", encoding="utf-8") as handle:
			for line in handle:
				line = line.strip()
				if not line:
					continue
				records.append(json.loads(line))
		if len(records) == 0:
			raise ValueError(f"Manifest is empty: {manifest_path}")
		return records

	def __len__(self) -> int:
		"""Return the active manifest subset size."""

		return len(self.records)

	def __getitem__(self, index: int) -> WorldModelSample:
		"""Load one sample from disk.

		Args:
			index: Dataset index.

		Returns:
			A fully materialized `WorldModelSample`.
		"""

		record = self.records[index]
		reference_paths = record["reference_images"][: self.config.num_reference_images]
		reference_images = [load_image(path) for path in reference_paths]
		video = load_video_frames(
			record["video_path"],
			num_frames=self.config.video_num_frames,
			stride=self.config.video_frame_stride,
			height=self.config.video_height,
			width=self.config.video_width,
			random_clip=self.config.shuffle,
		)
		return WorldModelSample(
			sample_id=str(record.get("id", index)),
			prompt=record["prompt"],
			reference_images=reference_images,
			video=video,
		)


class WebDatasetVideoCaptionDataset(IterableDataset):
	"""Temporary WebDataset loader for video-caption tar shards.

	Each grouped sample is expected to contain:

	- one `.mp4` file with the source video,
	- one `.json` metadata file containing `video_caption`.

	The prompt is read from `video_caption`, and reference images are sampled
	randomly from the decoded video frames.
	"""

	def __init__(self, config: DatasetConfig):
		"""Initialize the WebDataset-backed iterable dataset.

		Args:
			config: Dataset section of the root config.
		"""

		self.config = config
		if len(config.webdataset_urls) == 0:
			raise ValueError("`dataset.webdataset_urls` must be provided for `dataset_source=webdataset`.")
		if config.num_samples <= 0:
			raise ValueError("`dataset.num_samples` must be positive for `dataset_source=webdataset`.")
		self.shard_paths = self._resolve_shard_paths(config.webdataset_urls)

	def __len__(self) -> int:
		"""Return the exact capped sample count seen by one distributed process."""

		return self._rank_sample_limit()

	def _rank_sample_limit(self) -> int:
		"""Return this process's deterministic share of the global sample cap.

		WebDataset is sharded inside the dataset rather than by Accelerate's
		dataloader wrapper, so every distributed rank must consume the same number
		of samples to avoid uneven collective calls during training.
		"""

		world_size = get_world_size()
		if self.config.num_samples % world_size != 0:
			raise ValueError(
				"`dataset.num_samples` must be divisible by the distributed world size when "
				f"`dataset_source=webdataset`; got num_samples={self.config.num_samples}, world_size={world_size}."
			)
		return self.config.num_samples // world_size

	def _worker_sample_limit(self) -> int:
		"""Return this dataloader worker's deterministic share of the rank cap."""

		worker_info = get_worker_info()
		if worker_info is None:
			return self._rank_sample_limit()
		return partition_count(
			self._rank_sample_limit(),
			worker_info.id,
			worker_info.num_workers,
		)

	def _resolve_shard_paths(self, urls: list[str] | str) -> list[str]:
		"""Resolve the configured shard globs into a stable path list.

		Args:
			urls: One or more shard glob patterns.

		Returns:
			A sorted list of matching shard paths.

		Raises:
			ValueError: If none of the shard patterns match.
		"""

		patterns = urls if isinstance(urls, list) else [urls]
		shard_paths = sorted(path for pattern in patterns for path in glob.glob(pattern))
		if len(shard_paths) == 0:
			raise ValueError(f"No WebDataset shards matched: {patterns}")
		return shard_paths

	def _parse_metadata(self, metadata: bytes | str | dict[str, Any]) -> dict[str, Any]:
		"""Convert the stored metadata payload into a Python dictionary.

		Args:
			metadata: Raw `.json` payload from WebDataset.

		Returns:
			A parsed metadata dictionary.
		"""

		if isinstance(metadata, dict):
			return metadata
		if isinstance(metadata, bytes):
			return json.loads(metadata.decode("utf-8"))
		return json.loads(metadata)

	def _build_sample(self, sample: dict[str, Any]) -> WorldModelSample:
		"""Convert one grouped WebDataset sample into a `WorldModelSample`.

		Args:
			sample: Grouped sample dictionary produced by WebDataset.

		Returns:
			A fully materialized `WorldModelSample`.
		"""

		if "mp4" not in sample or "json" not in sample:
			raise ValueError("Each WebDataset sample must contain both `mp4` and `json` entries.")

		metadata = self._parse_metadata(sample["json"])
		raw_frames = decode_video_frames(sample["mp4"])
		sampled_frames = sample_video_frames_from_raw_frames(
			raw_frames,
			num_frames=self.config.video_num_frames,
			frame_sampling="uniform",
		)
		reference_images = sample_reference_images(
			sampled_frames,
			num_reference_images=self.config.num_reference_images,
			random_selection=True,
			preserve_order=False,
		)
		video = video_frames_to_tensor(
			sampled_frames,
			height=self.config.video_height,
			width=self.config.video_width,
		)
		return WorldModelSample(
			sample_id=str(sample.get("__key__", metadata.get("id", ""))),
			prompt=str(metadata["video_caption"]),
			reference_images=reference_images,
			video=video,
		)

	def __iter__(self):
		"""Iterate over a deterministic capped subset of grouped tar samples."""

		sample_limit = self._worker_sample_limit()
		if sample_limit <= 0:
			return
		dataset = wds.WebDataset(
			self.shard_paths,
			shardshuffle=100 if self.config.shuffle else False,
			nodesplitter=wds.split_by_node,
			workersplitter=wds.split_by_worker,
		)
		if self.config.shuffle:
			dataset = dataset.shuffle(1000)
		num_emitted = 0
		for sample in dataset:
			if num_emitted >= sample_limit:
				break
			yield self._build_sample(sample)
			num_emitted += 1


def build_dataset(config: DatasetConfig) -> Dataset | IterableDataset:
	"""Construct the requested dataset backend from config.

	Args:
		config: Dataset section of the root config.

	Returns:
		Either a map-style manifest dataset or an iterable WebDataset loader.
	"""

	if config.dataset_source == "manifest":
		return ManifestWorldModelDataset(config)
	if config.dataset_source == "webdataset":
		return WebDatasetVideoCaptionDataset(config)
	raise ValueError(f"Unsupported dataset source: {config.dataset_source}")


class WorldModelBatchCollator:
	"""Collate raw samples into the three model-specific input views.

	The collator prepares:

	- tokenized prompt text for Qwen,
	- flattened Qwen image inputs for visual encoding,
	- square VGGT image batches for geometry encoding,
	- padded video tensors for Wan VAE supervision.
	"""

	def __init__(self, dataset_config: DatasetConfig, tokenizer, image_processor, geo_patch_size: int):
		"""Create the collator.

		Args:
			dataset_config: Dataset section of the root config.
			tokenizer: Qwen tokenizer used for prompt text.
			image_processor: Qwen image processor used for reference images.
			geo_patch_size: VGGT patch size used to validate geometry image resolution.
		"""

		self.dataset_config = dataset_config
		self.tokenizer = tokenizer
		self.image_processor = image_processor
		self.geo_patch_size = geo_patch_size
		self.geo_image_size = resolve_geo_image_size(dataset_config, geo_patch_size)

	def _build_reference_batches(self, samples: list[WorldModelSample]) -> tuple[list[Any], Tensor, Tensor]:
		"""Prepare shared reference images for both Qwen and VGGT.

		Args:
			samples: Batch of raw samples.

		Returns:
			A tuple containing:
			- flattened resized PIL images for Qwen,
			- a tensor with shape `(B, S_ref, 3, H, W)` for VGGT,
			- a boolean mask indicating which reference slots are valid.
		"""

		max_refs = max(len(sample.reference_images) for sample in samples)
		vis_size = self.dataset_config.vis_image_size
		geo_size = self.geo_image_size
		vis_images_flat: list[Any] = []
		geo_images = torch.zeros(len(samples), max_refs, 3, geo_size, geo_size, dtype=torch.float32)
		reference_mask = torch.zeros(len(samples), max_refs, dtype=torch.bool)

		for batch_index, sample in enumerate(samples):
			for ref_index, image in enumerate(sample.reference_images):
				cropped_image = center_crop_to_aspect(image, vis_size, vis_size)
				vis_image = resize_image(cropped_image, vis_size, vis_size)
				geo_image = vis_image if geo_size == vis_size else resize_image(cropped_image, geo_size, geo_size)
				vis_images_flat.append(vis_image)
				geo_images[batch_index, ref_index] = pil_to_tensor(geo_image, normalize_to_neg_one=False)
				reference_mask[batch_index, ref_index] = True

		return vis_images_flat, geo_images, reference_mask

	def _pad_videos(self, samples: list[WorldModelSample]) -> tuple[Tensor, Tensor]:
		"""Pad variable-length videos to the longest clip in the batch.

		Args:
			samples: Batch of raw samples.

		Returns:
			A tuple of:
			- video tensor with shape `(B, 3, T_max, H, W)`,
			- per-sample valid frame counts before padding.
		"""

		max_frames = max(sample.video.size(0) for sample in samples)
		height = samples[0].video.size(-2)
		width = samples[0].video.size(-1)
		videos = torch.zeros(len(samples), 3, max_frames, height, width, dtype=torch.float32)
		frame_counts = torch.zeros(len(samples), dtype=torch.long)

		for batch_index, sample in enumerate(samples):
			video = sample.video
			num_frames = video.size(0)
			frame_counts[batch_index] = num_frames
			videos[batch_index, :, :num_frames] = video.permute(1, 0, 2, 3)
			if num_frames < max_frames:
				videos[batch_index, :, num_frames:] = video[-1:].permute(1, 0, 2, 3)

		return videos, frame_counts

	def __call__(self, samples: list[WorldModelSample]) -> dict[str, Tensor | list[str]]:
		"""Collate a list of raw samples into one training batch.

		Args:
			samples: Batch of dataset samples.

		Returns:
			A dictionary containing prompt tokens, Qwen image inputs, VGGT image
			inputs, video tensors, and the bookkeeping needed to regroup the flat
			reference-image features back per sample.
		"""

		prompts = [sample.prompt for sample in samples]
		vis_ref_counts = torch.tensor([len(sample.reference_images) for sample in samples], dtype=torch.long)

		txt_inputs = self.tokenizer(
			prompts,
			padding=True,
			truncation=True,
			max_length=self.dataset_config.max_text_length,
			return_tensors="pt",
		)
		vis_images_flat, geo_images, reference_mask = self._build_reference_batches(samples)
		vis_inputs = self.image_processor(images=vis_images_flat, do_resize=False, return_tensors="pt")
		videos, frame_counts = self._pad_videos(samples)

		return {
			"sample_ids": [sample.sample_id for sample in samples],
			"prompts": prompts,
			"txt_input_ids": txt_inputs["input_ids"],
			"txt_attention_mask": txt_inputs["attention_mask"],
			"qwen_vis_pixel_values": vis_inputs["pixel_values"],
			"qwen_vis_grid_thw": torch.as_tensor(vis_inputs["image_grid_thw"], dtype=torch.long),
			"vis_ref_counts": vis_ref_counts,
			"geo_images": geo_images,
			"reference_mask": reference_mask,
			"videos": videos,
			"video_frame_counts": frame_counts,
		}
