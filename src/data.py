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
	center_crop_and_resize,
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


class DeepWorldQWBatchCollator:
	"""Collate raw samples into the Qwen/Wan model-specific input views.

	DeepWorldQW trains one sample at a time. The collator prepares that sample's
	reference-image, prompt, and video views without adding a sample batch axis.
	"""

	def __init__(self, dataset_config: DatasetConfig, tokenizer, image_processor, geo_patch_size: int):
		"""Create the QW collator.

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

	def _build_reference_views(self, sample: WorldModelSample) -> tuple[list[Any], Tensor]:
		"""Prepare one sample's reference images for Qwen and VGGT."""

		reference_count = len(sample.reference_images)
		vis_size = self.dataset_config.vis_image_size
		geo_size = self.geo_image_size
		vis_images_flat: list[Any] = []
		geo_images = torch.empty(reference_count, 3, geo_size, geo_size, dtype=torch.float32)

		for ref_index, image in enumerate(sample.reference_images):
			cropped_image = center_crop_to_aspect(image, vis_size, vis_size)
			vis_image = resize_image(cropped_image, vis_size, vis_size)
			geo_image = vis_image if geo_size == vis_size else resize_image(cropped_image, geo_size, geo_size)
			vis_images_flat.append(vis_image)
			geo_images[ref_index] = pil_to_tensor(geo_image, normalize_to_neg_one=False)

		return vis_images_flat, geo_images

	def __call__(self, samples: list[WorldModelSample]) -> dict[str, Tensor | str]:
		"""Collate one raw sample into QW training inputs.

		Args:
			samples: Local dataloader sample list. QW keeps PyTorch
				`batch_size=1` for Accelerate compatibility and unwraps it here.

		Returns:
			A dictionary containing prompt tokens, Qwen image inputs, VGGT image
			inputs, and video tensors.
		"""

		sample = samples[0]
		txt_inputs = self.tokenizer(
			sample.prompt,
			padding=False,
			truncation=True,
			max_length=self.dataset_config.max_text_length,
			return_tensors="pt",
		)
		vis_images_flat, geo_images = self._build_reference_views(sample)
		vis_inputs = self.image_processor(images=vis_images_flat, do_resize=False, return_tensors="pt")
		video = sample.video.permute(1, 0, 2, 3).contiguous()

		return {
			"sample_id": sample.sample_id,
			"prompt": sample.prompt,
			"txt_input_ids": txt_inputs["input_ids"][0],
			"txt_attention_mask": txt_inputs["attention_mask"][0],
			"qwen_vis_pixel_values": vis_inputs["pixel_values"],
			"qwen_vis_grid_thw": torch.as_tensor(vis_inputs["image_grid_thw"], dtype=torch.long),
			"geo_images": geo_images,
			"video": video,
		}


class DeepWorldHYBatchCollator:
	"""Collate curated samples for the Hunyuan/VGGT three-stream world model.

	The Hunyuan path keeps reference-image views separate because each frozen
	encoder has different resolution and value-range expectations:

	- `vis_ref_images`: SigLIP-sized tensors in `[0, 1]`,
	- `geo_ref_images`: VGGT-sized tensors in `[0, 1]`,
	- `vae_ref_images`: Hunyuan-VAE-sized tensors in `[-1, 1]`.

	The training dataloader always supplies one sample at a time, so no
	cross-sample reference or video padding is needed and no sample batch axis is
	added here.
	"""

	def __init__(self, dataset_config: DatasetConfig, geo_patch_size: int):
		"""Create the Hunyuan collator.

		Args:
			dataset_config: Dataset section of the root config.
			geo_patch_size: VGGT patch size used to validate geometry image resolution.
		"""

		self.dataset_config = dataset_config
		self.geo_patch_size = geo_patch_size
		self.geo_image_size = resolve_geo_image_size(dataset_config, geo_patch_size)

	def _build_reference_views(
		self,
		sample: WorldModelSample,
	) -> tuple[Tensor, Tensor, Tensor]:
		"""Prepare one sample's reference-image tensors for all frozen encoders."""

		reference_count = len(sample.reference_images)
		vis_size = self.dataset_config.vis_image_size
		geo_size = self.geo_image_size
		vae_height = self.dataset_config.video_height
		vae_width = self.dataset_config.video_width

		vis_ref_images = torch.empty(reference_count, 3, vis_size, vis_size, dtype=torch.float32)
		geo_ref_images = torch.empty(reference_count, 3, geo_size, geo_size, dtype=torch.float32)
		vae_ref_images = torch.empty(reference_count, 3, vae_height, vae_width, dtype=torch.float32)

		for ref_index, image in enumerate(sample.reference_images):
			semantic_image = center_crop_and_resize(image, vis_size, vis_size)
			geo_image = center_crop_and_resize(image, geo_size, geo_size)
			vae_image = center_crop_and_resize(image, vae_height, vae_width)

			vis_ref_images[ref_index] = pil_to_tensor(semantic_image, normalize_to_neg_one=False)
			geo_ref_images[ref_index] = pil_to_tensor(geo_image, normalize_to_neg_one=False)
			vae_ref_images[ref_index] = pil_to_tensor(vae_image, normalize_to_neg_one=True)

		return vis_ref_images, geo_ref_images, vae_ref_images

	def __call__(self, sample: WorldModelSample) -> dict[str, Tensor | str]:
		"""Collate one raw sample into Hunyuan world-model training inputs."""

		vis_ref_images, geo_ref_images, vae_ref_images = self._build_reference_views(sample)
		video = sample.video.permute(1, 0, 2, 3).contiguous()

		return {
			"sample_id": sample.sample_id,
			"prompt": sample.prompt,
			"vis_ref_images": vis_ref_images,
			"geo_ref_images": geo_ref_images,
			"vae_ref_images": vae_ref_images,
			"video": video,
		}
