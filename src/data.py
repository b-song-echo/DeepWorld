import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.config import DatasetConfig
from src.utils import (
	center_crop_to_aspect,
	load_image,
	load_video_frames,
	pil_to_tensor,
	resize_image,
)


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


class WorldDataset(Dataset):
	"""Curated ScanNet++ world-model dataset backed by a JSONL manifest.

	Each manifest row is produced by `data_gen.py` and contains a self-contained
	ground-truth clip, ordered reference images, a rich synthesized prompt, and
	medium/coarse distilled prompt variants. Reference order is preserved because
	the prompt may refer to "the first image" or similar fixed indices.
	"""

	def __init__(self, config: DatasetConfig):
		"""Initialize the dataset and load the manifest.

		Args:
			config: Dataset section of the root config.
		"""

		if not config.manifest_path:
			raise ValueError("`dataset.manifest_path` must point to a curated JSONL manifest.")
		self.config = config
		self.data_root = Path(config.data_root) if config.data_root else Path(config.manifest_path).parent
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

	def _resolve_path(self, path: str) -> Path:
		"""Resolve one manifest path against the configured dataset root.

		Args:
			path: Absolute path or manifest-relative sample path.

		Returns:
			A concrete local filesystem path.
		"""

		candidate = Path(path)
		if candidate.is_absolute():
			return candidate
		return self.data_root / candidate

	def _choose_prompt(self, record: dict[str, Any]) -> str:
		"""Sample one prompt granularity according to configured probabilities.

		Args:
			record: Parsed manifest row containing rich and distilled prompts.

		Returns:
			The selected prompt string.
		"""

		distilled = record.get("distilled_prompts") or {}
		prompt_table = [
			(str(record["synthesized_prompt"]), self.config.prompt_rich_prob),
			(str(distilled.get("medium") or record["synthesized_prompt"]), self.config.prompt_medium_prob),
			(str(distilled.get("coarse") or distilled.get("medium") or record["synthesized_prompt"]), self.config.prompt_coarse_prob),
		]
		total = sum(weight for _, weight in prompt_table)
		draw = random.random() * total
		running = 0.0
		for prompt, weight in prompt_table:
			running += weight
			if draw <= running:
				return prompt
		return prompt_table[-1][0]

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
		reference_images = [
			load_image(self._resolve_path(ref["path"]))
			for ref in record["ref_imgs"]
		]
		video = load_video_frames(
			self._resolve_path(record["gt_clip"]["path"]),
			num_frames=self.config.video_num_frames,
			height=self.config.video_height,
			width=self.config.video_width,
			random_clip=False,
			frame_sampling="uniform",
		)
		return WorldModelSample(
			sample_id=str(record.get("sample_id", index)),
			prompt=self._choose_prompt(record),
			reference_images=reference_images,
			video=video,
		)


def build_dataset(config: DatasetConfig) -> Dataset:
	"""Construct the curated world-model dataset from config.

	Args:
		config: Dataset section of the root config.

	Returns:
		A map-style curated-manifest dataset.
	"""

	return WorldDataset(config)


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
