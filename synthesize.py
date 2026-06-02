from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import ffmpeg
import imageio.v3 as iio
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info
import torch
from filelock import FileLock
from transformers import (
	AutoModelForCausalLM,
	AutoModelForImageTextToText,
	AutoProcessor,
	AutoTokenizer
)

from scannetpp.common.scene_release import ScannetppScene_Release
from src.prompts import (
	CAPTION_REPHRASING_TEMPLATE,
	CAPTION_WIRING_TEMPLATE,
	CRITIC_JUDGING_TEMPLATE,
	DISTILLATION_TEMPLATE,
	IMAGE_CAPTIONING_TEMPLATE,
	MOTION_DIGESTING_TEMPLATE,
	VIDEO_CAPTIONING_TEMPLATE,
)


class RejectedSample(Exception):
	"""Expected rejection for a sampled candidate that fails a curation gate."""


@dataclass
class RefSource:
	"""One selected reference source before it is copied into the sample."""

	kind: str
	ref_id: str
	image_path: Path | None
	mask_path: Path | None
	is_start_frame: bool


@dataclass
class SampleContext:
	"""Mutable state for one candidate sample while it moves through stages."""

	scene_id: str = ""
	scene_type: str = "unknown"
	sample_id: str = ""
	start_frame: int = 0
	clip_frames: int = 0
	clip_duration_s: float = 0.0
	video_width: int = 0
	video_height: int = 0
	video_fps: float = 0.0
	video_num_frames: int = 0
	video_duration_s: float = 0.0
	source_video_path: Path | None = None
	source_video_mask_path: Path | None = None
	source_pose_path: Path | None = None
	tmp_dir: Path | None = None
	final_dir: Path | None = None
	manifest_entry: dict[str, Any] = field(default_factory=dict)

	@property
	def intermediate_dir(self) -> Path:
		"""Return the intermediate-output directory for this sample."""

		return self.require_tmp_dir() / "intermediate"

	@property
	def gt_clip_path(self) -> Path:
		"""Return the temporary ground-truth clip path."""

		return self.require_tmp_dir() / "gt_clip.mp4"

	@property
	def ref_img_dir(self) -> Path:
		"""Return the temporary reference-image directory."""

		return self.require_tmp_dir() / "ref_imgs"

	@property
	def video_square_size(self) -> int:
		"""Return the side length of the center-cropped square video."""

		return min(self.video_width, self.video_height)

	def require_tmp_dir(self) -> Path:
		"""Return the temporary sample directory or raise if it is unset."""

		if self.tmp_dir is None:
			raise RuntimeError("Temporary sample directory is not initialized.")
		return self.tmp_dir

	def require_final_dir(self) -> Path:
		"""Return the final sample directory or raise if it is unset."""

		if self.final_dir is None:
			raise RuntimeError("Final sample directory is not initialized.")
		return self.final_dir

	def require_source_video_path(self) -> Path:
		"""Return the source RGB video path or raise if it is unset."""

		if self.source_video_path is None:
			raise RuntimeError("Source video path is not initialized.")
		return self.source_video_path

	def require_source_video_mask_path(self) -> Path:
		"""Return the source video mask path or raise if it is unset."""

		if self.source_video_mask_path is None:
			raise RuntimeError("Source video mask path is not initialized.")
		return self.source_video_mask_path

	def require_source_pose_path(self) -> Path:
		"""Return the source pose path or raise if it is unset."""

		if self.source_pose_path is None:
			raise RuntimeError("Source pose path is not initialized.")
		return self.source_pose_path


def parse_args() -> Namespace:
	"""Parse the offline command line."""

	parser = ArgumentParser(description="Generate curated DeepWorld samples from ScanNet++ scenes.")
	parser.add_argument("--scannetpp_root", type=str, required=True)
	parser.add_argument("--output_root", type=str, required=True)
	parser.add_argument("--num_processes", type=int, default=1)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--split", type=str, choices=["train", "val"], required=True)
	parser.add_argument("--restart_fresh", action="store_true")
	parser.add_argument("--no_cleanup", action="store_true")
	parser.add_argument("--num_samples", type=int, required=True)
	parser.add_argument("--clip_seconds", type=float, default=5.0)
	parser.add_argument("--max_ref_images", type=int, default=8)
	parser.add_argument("--include_start_frame_prob", type=float, default=0.4)
	parser.add_argument("--vlm_backend_path", type=str, required=True)
	parser.add_argument("--llm_backend_path", type=str, required=True)
	parser.add_argument("--vlm_cpu_offload", action="store_true")
	parser.add_argument("--llm_cpu_offload", action="store_true")
	parser.add_argument("--llm_think_if_available", action="store_true")
	parser.add_argument("--video_captioning_width", type=int, default=720)
	parser.add_argument("--video_captioning_height", type=int, default=720)
	parser.add_argument("--video_captioning_fps", type=float, default=2.0)
	parser.add_argument("--video_captioning_vlm_temperature", type=float, default=0.2)
	parser.add_argument("--video_captioning_vlm_max_new_tokens", type=int, default=2048)
	parser.add_argument("--image_captioning_width", type=int, default=720)
	parser.add_argument("--image_captioning_height", type=int, default=720)
	parser.add_argument("--image_captioning_vlm_temperature", type=float, default=0.2)
	parser.add_argument("--image_captioning_vlm_max_new_tokens", type=int, default=1024)
	parser.add_argument("--motion_digesting_unit_seconds", type=float, default=1.0)
	parser.add_argument("--motion_digesting_llm_temperature", type=float, default=0.1)
	parser.add_argument("--motion_digesting_llm_max_new_tokens", type=int, default=2048)
	parser.add_argument("--caption_wiring_llm_temperature", type=float, default=0.2)
	parser.add_argument("--caption_wiring_llm_max_new_tokens", type=int, default=2048)
	parser.add_argument("--caption_rephrasing_llm_temperature", type=float, default=0.4)
	parser.add_argument("--caption_rephrasing_llm_max_new_tokens", type=int, default=1536)
	parser.add_argument("--critic_judging_llm_temperature", type=float, default=0.0)
	parser.add_argument("--critic_judging_llm_max_new_tokens", type=int, default=2048)
	parser.add_argument("--distillation_llm_temperature", type=float, default=0.25)
	parser.add_argument("--distillation_llm_max_new_tokens", type=int, default=1024)
	parser.add_argument("--filter_pixel_valid_fraction_min", type=float, default=None)
	parser.add_argument("--filter_pose_valid_fraction_min", type=float, default=None)
	parser.add_argument("--filter_camera_trajectory_length_m_min", type=float, default=None)
	parser.add_argument("--filter_camera_trajectory_length_m_max", type=float, default=None)
	parser.add_argument("--filter_quality_score_min", type=float, default=None)
	
	args = parser.parse_args()
	if args.num_processes < 1:
		raise ValueError(f"`--num_processes` must be positive, got {args.num_processes}.")
	if args.num_samples < 0:
		raise ValueError(f"`--num_samples` must be non-negative, got {args.num_samples}.")
	if args.restart_fresh and args.no_cleanup:
		raise ValueError("`--restart_fresh` and `--no_cleanup` cannot be used together.")
	if args.clip_seconds <= 0:
		raise ValueError(f"`--clip_seconds` must be positive, got {args.clip_seconds}.")
	if args.max_ref_images < 1:
		raise ValueError(f"`--max_ref_images` must be at least 1, got {args.max_ref_images}.")
	if not 0.0 <= args.include_start_frame_prob <= 1.0:
		raise ValueError("`--include_start_frame_prob` must be in [0, 1].")
	if args.motion_digesting_unit_seconds <= 0:
		raise ValueError("`--motion_digesting_unit_seconds` must be positive.")
	return args


def output_root(args: Namespace) -> Path:
	"""Return the output root path."""

	return Path(args.output_root)


def manifest_path(args: Namespace) -> Path:
	"""Return the split-specific manifest path."""

	return output_root(args) / f"manifest_{args.split}.jsonl"


def state_path(args: Namespace) -> Path:
	"""Return the split-specific shared state path."""

	return output_root(args) / f"state_{args.split}.json"


def exclusive_lock(args: Namespace) -> FileLock:
	"""Return the global runtime lock that guards manifest and state updates together."""

	path = output_root(args) / "runtime_lock.lock"
	path.parent.mkdir(parents=True, exist_ok=True)
	return FileLock(str(path))


def read_json(path: Path) -> Any:
	"""Read one JSON file."""

	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
	"""Write one formatted JSON file."""

	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, ensure_ascii=False)
		f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
	"""Read one JSONL into entry dictionaries."""

	if not path.exists():
		return []
	entries: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as f:
		for i, line in enumerate(f, start=1):
			if not line.strip():
				continue
			entry = json.loads(line)
			if not isinstance(entry, dict):
				raise ValueError(f"Manifest entry {path}:{i} is not a JSON object.")
			entries.append(entry)
	return entries


def write_jsonl(
	path: Path,
	*payload: list[dict[str, Any]],
	append=True
) -> None:
	"""Write/append JSONL entries and fsync it."""

	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a" if append else "w", encoding="utf-8", buffering=1) as f:
		for entry in payload:
			f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
		f.flush()
		os.fsync(f.fileno())


class DataSamplingStage:
	"""Sample ScanNet++ media and write one candidate artifact.

	This stage owns all filesystem interactions with ScanNet++ scene assets. It
	selects a split-appropriate scene, extracts the square iPhone clip, prepares
	a fixed shuffled order of reference images, computes pixel-mask fractions,
	and writes the sample-local media files under `samples/.tmp_<sample_id>/`.
	Downstream stages only read this self-contained temporary sample directory.
	"""

	def __init__(
		self,
		args: Namespace,
		rng: random.Random,
	):
		self.args = args
		self.scene_ids = self._load_scene_ids()
		self.scene_types = self._load_scene_types()
		self.rng = rng
		self.data_root = Path(args.scannetpp_root)
		self.samples_root = output_root(args) / "samples"
	
	def _load_scene_types(self) -> dict[str, str]:
		path = Path(self.args.scannetpp_root) / "metadata" / "scene_types.json"
		if not path.exists():
			return {}
		return read_json(path)

	def _load_scene_ids(self) -> list[str]:
		split_name = f"nvs_sem_{self.args.split}.txt"
		path = Path(self.args.scannetpp_root) / "splits" / split_name
		if not path.exists():
			raise FileNotFoundError(f"ScanNet++ split file not found: {path}")
		with path.open("r", encoding="utf-8") as handle:
			scene_ids = [line.strip() for line in handle if line.strip()]
		if len(scene_ids) == 0:
			raise ValueError(f"ScanNet++ split file is empty: {path}")
		return scene_ids

	def _probe_video(self, ctx: SampleContext) -> None:
		"""Probe source video stream metadata into the sample context."""

		def parse_duration(value: Any) -> float | None:
			if value is None:
				return None
			text = str(value).strip()
			if not text or text == "N/A":
				return None
			if ":" not in text:
				try:
					duration = float(text)
				except ValueError:
					return None
				return duration if duration > 0 else None

			parts = text.split(":")
			if len(parts) != 3:
				return None
			try:
				hours, minutes, seconds = parts
				duration = int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)
			except ValueError:
				return None
			return duration if duration > 0 else None
		
		def parse_frame_rate(value: Any) -> float | None:
			if value is None:
				return None
			try:
				fps = float(Fraction(str(value)))
			except (ValueError, ZeroDivisionError):
				return None
			return fps if fps > 0 else None

		def parse_num_frames(value: Any) -> int:
			if value is None:
				return 0
			try:
				return min(int(value), 0)
			except ValueError:
				return 0
		
		path = ctx.require_source_video_path()
		probe = ffmpeg.probe(str(path), select_streams="v:0")
		streams = probe.get("streams") or []
		if not streams:
			raise RejectedSample(f"No video stream found in {path}")
		stream = streams[0]

		duration = (
			parse_duration(stream.get("duration"))
			or parse_duration((stream.get("tags") or {}).get("DURATION"))
			or parse_duration((probe.get("format") or {}).get("duration"))
		)

		frame_rate = (
			parse_frame_rate(stream.get("avg_frame_rate"))
			or parse_frame_rate(stream.get("r_frame_rate"))
		)
		if frame_rate is None:
			raise RejectedSample(f"Could not determine FPS for video: {path}")
			
		num_frames = (
			parse_num_frames(stream.get("nb_frames"))
			or int(round(frame_rate * (duration or 0.0)))
		)
		if num_frames <= 0:
			raise RejectedSample(f"Could not determine frame count for video: {path}")

		ctx.video_width = int(stream["width"])
		ctx.video_height = int(stream["height"])
		ctx.video_fps = frame_rate
		ctx.video_num_frames = num_frames
		ctx.video_duration_s = duration if duration is not None else num_frames / frame_rate
	
	def _prepare_gt_clip(self, ctx: SampleContext) -> None:
		"""Extract and center-crop one clip while preserving the source FPS."""

		def center_crop_args() -> tuple[str, str, str, str]:
			side = "min(iw,ih)"
			return side, side, f"(iw-{side})/2", f"(ih-{side})/2"
		
		source_video_path = ctx.require_source_video_path()
		if not source_video_path.exists():
			raise RejectedSample(f"Missing iPhone video: {source_video_path}")
		source_mask_path = ctx.require_source_video_mask_path()
		if source_mask_path.exists():
			output_kwargs = {"format": "rawvideo", "pix_fmt": "gray"}
			process = (
				ffmpeg.input(
					str(source_mask_path),
					ss=ctx.start_frame / ctx.video_fps, t=self.args.clip_seconds
				).video
				.filter_("crop", *center_crop_args())
				.filter_("fps", fps=ctx.video_fps)
				.filter_("format", "gray")
				.output("pipe:", **output_kwargs)
				.run_async(pipe_stdout=True, pipe_stderr=True)
			)
			assert process.stdout is not None
			frame_size = ctx.video_square_size ** 2
			valid_pixels, total_pixels = 0, 0
			while True:
				chunk = process.stdout.read(frame_size)
				if not chunk:
					break
				if len(chunk) != frame_size:
					break
				mask = np.frombuffer(chunk, dtype=np.uint8)
				valid_pixels += int((mask > 127).sum())
				total_pixels += mask.size
			_, stderr = process.communicate()
			if process.returncode not in {0, None}:
				raise RuntimeError(stderr.decode("utf-8", errors="replace"))
			if total_pixels > 0:
				video_valid_fraction = valid_pixels / total_pixels
				if self.args.filter_pixel_valid_fraction_min is not None and video_valid_fraction < self.args.filter_pixel_valid_fraction_min:
					raise RejectedSample(f"Video valid fraction {video_valid_fraction:.4f} below threshold.")

		try:
			process = (
				ffmpeg
				.input(
					str(source_video_path),
					ss=ctx.start_frame / ctx.video_fps, t=self.args.clip_seconds
				).video
				.filter_("crop", *center_crop_args())
				.filter_("fps", fps=ctx.video_fps)
				.output(
					str(ctx.gt_clip_path),
					vcodec="libx264", crf=18, pix_fmt="yuv420p", an=None,
				).overwrite_output()
				.run(capture_stdout=True, capture_stderr=True)
			)
			ctx.manifest_entry["gt_clip"] = {
				"fps": ctx.video_fps,
				"duration_sec": self.args.clip_seconds,
				"width": ctx.video_square_size,
				"height": ctx.video_square_size,
				"path": f"samples/{ctx.sample_id}/gt_clip.mp4",
			}
		except ffmpeg.Error as error:
			message = error.stderr.decode("utf-8", errors="replace") if error.stderr else str(error)
			raise RuntimeError(f"ffmpeg failed while writing {ctx.gt_clip_path}: {message}") from error

	# TODO: This method should parse the raw file to a clean python list, where each element is either None or a nested list and can be converted directly to numpy array. Then it store the slice to `intermediate/poses.json`.
	def _prepare_poses(self, ctx: SampleContext) -> None:
		"""Load aligned iPhone poses from the supported pose JSON layouts."""

		def sort_key(key: Any) -> tuple[int, str]:
			text = str(key)
			if text.isdigit():
				return int(text), text
			stem = Path(text).stem
			digits = "".join(character for character in stem if character.isdigit())
			if digits:
				return int(digits), text
			return 10**12, text

		def parse_payload(poses: Any) -> list[Any]:
			if poses is None:
				return []
			if isinstance(poses, dict):				
				items = sorted(poses.items(), key=lambda item: sort_key(item[0]))
				return [value for _, value in items]
			return list(poses)
		
		def parse_pose(value: Any) -> np.ndarray | None:
			if value is None:
				return None
			if isinstance(value, dict):
				fields = ("aligned_pose", "transform_matrix", "pose", "c2w")
				value = next((value[field] for field in fields if field in value), None)
			try:
				array = np.asarray(value, dtype=np.float64)
			except (TypeError, ValueError):
				return None
			if array.shape != (4, 4) or not np.isfinite(array).all():
				return None
			return array

		path = ctx.require_source_pose_path()
		if not path.exists():
			raise RejectedSample(f"Missing iPhone pose file: {path}")
		payload = read_json(path)
		poses = (
			parse_payload(payload.get("aligned_poses"))
			or parse_payload(payload.get("poses"))
			or parse_payload(payload)
		) if isinstance(payload, dict) else parse_payload(payload)
		if not poses:
			raise RejectedSample(f"No pose records found in {path}")
		# TODO: Continue implementing.
		poses = [parse_pose(pose) for pose in poses]
		clip_poses = poses[ctx.start_frame:(ctx.start_frame + ctx.clip_frames)]		
		if len(clip_poses) < ctx.clip_frames:
			raise RejectedSample("Pose sequence is shorter than the sampled clip.")	

	def _prepare_reference_images(
		self, ctx: SampleContext,
		scene: ScannetppScene_Release
	) -> None:
		"""Copy selected references into fixed sample order."""

		def im_read(video_path: Path, index: int) -> Image.Image:
			return Image.fromarray(iio.imread(video_path, index=index))
		
		def square_crop(image: Image.Image) -> Image.Image:
			side = min(image.width, image.height)
			left, top = (image.width - side) // 2, (image.height - side) // 2
			return image.crop((left, top, left + side, top + side))
		
		def square_rgb(image: Image.Image) -> Image.Image:
			return square_crop(image.convert("RGB"))
		
		def square_gray(image: Image.Image) -> Image.Image:
			return square_crop(image.convert("L"))
		
		def ensure_valid_fraction(mask: Image.Image):
			mask = np.asarray(mask)
			if mask.ndim == 3:
				mask = np.max(mask[..., :3], axis=-1)
			valid_fraction = float((mask > 127).mean())
			if self.args.filter_pixel_valid_fraction_min is not None and valid_fraction < self.args.filter_pixel_valid_fraction_min:
				raise RejectedSample(f"Reference valid fraction {valid_fraction:.4f} below threshold.")
		
		def resolve_dslr_path(file_value: str| None) -> Path | None:
			if not file_path:
				return None
			file_path = Path(file_value)
			for candidate in [
				file_path,
				scene.dslr_dir / file_path,
				scene.dslr_resized_undistorted_dir / file_path.name,
			]:
				if candidate.exists():
					return candidate
			return None
		
		ref_sources: list[dict[str, Any]] = []
		include_start = self.rng.random() < self.args.include_start_frame_prob
		if include_start:
			start_mask_path = ctx.require_source_video_mask_path()
			if start_mask_path.exists():
				start_mask = im_read(start_mask_path, ctx.start_frame)
				ensure_valid_fraction(square_gray(start_mask))
			start_image = square_rgb(im_read(ctx.gt_clip_path, 0))
			ref_sources.append({
				"image": start_image,
				"id": f"iphone:{ctx.start_frame:07d}",
			})
		
		dslr_transforms_path = scene.dslr_nerfstudio_transform_undistorted_path
		if not dslr_transforms_path.exists():
			raise RejectedSample(f"Missing DSLR transforms: {dslr_transforms_path}")
		payload = read_json(dslr_transforms_path)
		dslr_images = list(payload.get("frames", [])) + list(payload.get("test_frames", []))
		
		dslr_count = 0
		self.rng.shuffle(dslr_images)
		dslr_needed = self.rng.randint(1, self.args.max_ref_images) - int(include_start)
		while dslr_count < dslr_needed:
			if not dslr_images:
				raise RejectedSample(f"Scene {ctx.scene_id} DSLR images not enough.")
			dslr_image = dslr_images.pop()
			image_path = resolve_dslr_path(dslr_image.get("file_path"))
			mask_path = resolve_dslr_path(dslr_image.get("mask_path"))
			if not image_path or dslr_image.get("is_bad"):
				continue
			if mask_path:
				with Image.open(mask_path) as loaded:
					ensure_valid_fraction(square_gray(loaded))
			with Image.open(image_path) as loaded:
				image = square_rgb(loaded.copy())
			# TODO: Use the transform information to estimate its relavent to the video clip. Read from disk to get video poses.
			# TODO: Apply image quality related filters here.
			ref_sources.append({
				"image": image,
				"id": f"dslr:{image_path.name}",
			})
			dslr_count += 1
		
		ref_dir = ctx.ref_img_dir
		ref_dir.mkdir(parents=True, exist_ok=True)
		self.rng.shuffle(ref_sources)
		metadata: list[dict[str, Any]] = []
		for index, source in enumerate(ref_sources, start=1):
			image: Image.Image = source["image"]
			output_name = f"ref_{index:03d}.jpg"
			image.save(ref_dir / output_name, quality=95)
			metadata.append({
				"index": index,
				"is_start_frame": image["id"].startswith("iphone"),
				"width": image.width,
				"height": image.height,
				"path": f"samples/{ctx.sample_id}/ref_imgs/{output_name}",
			})
		
		payload = "|".join(source["id"] for source in ref_sources)
		ref_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
		ctx.sample_id = f"{ctx.scene_id}__f{ctx.start_frame:07d}__r{ref_hash}"
		ctx.manifest_entry["ref_imgs"] = metadata

	def __call__(self, ctx: SampleContext) -> None:
		"""Create one sampled candidate under `samples/.tmp_<sample_id>`."""
		
		ctx.scene_id = self.rng.choice(self.scene_ids)
		scene = ScannetppScene_Release(ctx.scene_id, data_root=self.data_root)
		ctx.scene_type = str(self.scene_types.get(ctx.scene_id, "unknown"))
		ctx.clip_duration_s = self.args.clip_seconds
		ctx.source_video_path = scene.iphone_video_path
		ctx.source_video_mask_path = scene.iphone_video_mask_path
		ctx.source_pose_path = scene.iphone_pose_intrinsic_imu_path
		self._probe_video(ctx)
		ctx.clip_frames = int(round(ctx.video_fps * self.args.clip_seconds))
		if ctx.clip_frames <= 1 or ctx.video_num_frames <= ctx.clip_frames:
			raise RejectedSample(f"Video too short for {self.args.clip_seconds}s clip: {ctx.source_video_path}")
		ctx.start_frame = self.rng.randint(0, ctx.video_num_frames - ctx.clip_frames)
		
		ctx.tmp_dir = self.samples_root / f".tmp_{ctx.sample_id}"
		ctx.final_dir = self.samples_root / ctx.sample_id
		if ctx.final_dir.exists() or ctx.tmp_dir.exists():
			raise RejectedSample(f"Sample already exists or is in progress: {ctx.sample_id}")
		ctx.tmp_dir.mkdir(parents=True)
		ctx.intermediate_dir.mkdir()
		ctx.manifest_entry = {
			"sample_id": ctx.sample_id,
			"scene_id": ctx.scene_id,
			"scene_type": ctx.scene_type,
			"split": self.args.split,
			"ref_imgs": [],
			"gt_clip": {},
			"synthesized_prompt": "",
			"distilled_prompts": {},
		}
		try:
			self._prepare_gt_clip(ctx)
			self._prepare_poses(ctx)
			self._prepare_reference_images(ctx, scene)
		except Exception:
			cleanup_sample_tmp(ctx)
			raise


class MotionExtractionStage:
	"""Extract numeric camera-motion evidence from iPhone poses.

	This stage reads ScanNet++ iPhone camera-to-world poses, slices the same
	temporal window as the sampled video clip, validates whole-clip pose
	coverage, and converts poses into metric camera-motion statistics. It writes
	`motion_extraction.json`, which contains whole-clip motion plus fixed-duration
	motion units consumed by the later language model digesting stage.
	"""

	def __init__(self, args: Namespace):
		self.args = args

	def _parse_rotation(self, rotation: np.ndarray) -> tuple[float, float, float]:
		"""Return approximate yaw, pitch, and roll in degrees for camera axes."""

		forward = rotation @ np.array([0.0, 0.0, 1.0])
		down = rotation @ np.array([0.0, 1.0, 0.0])
		yaw = math.degrees(math.atan2(forward[0], forward[2]))
		pitch = math.degrees(math.atan2(-forward[1], math.hypot(forward[0], forward[2])))
		roll = math.degrees(math.atan2(-down[0], down[1]))
		return yaw, pitch, roll

	# NOTE: This method should stay in this stage, it is responsible for returning valid poses of a clip or a unit, and reject the sample if the fraction drops bellow the specified threshold. DataSamplingStage does not filter valid fraction for poses.
	def _valid_clip_poses(self, poses: list[np.ndarray | None]) -> list[np.ndarray]:
		"""Return valid whole-clip poses."""
		
		valid_poses = [pose for pose in poses if pose is not None]
		if len(valid_poses) < 2:
			raise RejectedSample("Fewer than two valid poses in sampled clip.")
		valid_fraction = len(valid_poses) / len(poses)
		if self.args.filter_pose_valid_fraction_min is not None and valid_fraction < self.args.filter_pose_valid_fraction_min:
			raise RejectedSample(f"Pose valid fraction {valid_fraction:.4f} below threshold.")
		return valid_poses
	
	def _round(self, value: float) -> float:
		return round(float(value), 4)
	
	def _motion_stats(
		self,
		poses: list[np.ndarray],
		duration_s: float,
		prefix: str = "",
	) -> dict[str, float]:
		"""Compute translation, rotation, and path-length statistics for poses."""

		positions = [pose[:3, 3] for pose in poses]
		trajectory_length = sum(float(np.linalg.norm(positions[index] - positions[index - 1])) for index in range(1, len(positions)))
		relative = np.linalg.inv(poses[0]) @ poses[-1]
		translation = relative[:3, 3]
		yaw, pitch, roll = self._parse_rotation(relative[:3, :3])
		return {
			"duration_s": self._round(duration_s),
			f"{prefix}trajectory_length_m": self._round(trajectory_length),
			f"{prefix}translation_right_m": self._round(translation[0]),
			f"{prefix}translation_down_m": self._round(translation[1]),
			f"{prefix}translation_forward_m": self._round(translation[2]),
			f"{prefix}translation_distance_m": self._round(float(np.linalg.norm(translation))),
			f"{prefix}delta_yaw_deg": self._round(yaw),
			f"{prefix}delta_pitch_deg": self._round(pitch),
			f"{prefix}delta_roll_deg": self._round(roll),
		}

	def __call__(self, ctx: SampleContext) -> None:
		"""Compute and save `motion_extraction.json`."""

		# TODO: Here, clip_poses is obtained by reading `intermediate/poses.json`, then each of its element is converted to numpy array for downstream computation.
		clip_poses = []
		clip_valid_poses = self._valid_clip_poses(clip_poses)
		
		trajectory = self._motion_stats(
			poses=clip_valid_poses,
			duration_s=self.args.clip_seconds,
		)
		# TODO: More motion filters.
		trajectory_length = trajectory["trajectory_length_m"]
		if self.args.filter_camera_trajectory_length_m_min is not None and trajectory_length < self.args.filter_camera_trajectory_length_m_min:
			raise RejectedSample(f"Trajectory length {trajectory_length:.4f} below minimum.")
		if self.args.filter_camera_trajectory_length_m_max is not None and trajectory_length > self.args.filter_camera_trajectory_length_m_max:
			raise RejectedSample(f"Trajectory length {trajectory_length:.4f} above maximum.")

		first_pose = clip_valid_poses[0]
		motion_units: list[dict[str, Any]] = []
		unit = self.args.motion_digesting_unit_seconds
		num_units = int(math.ceil(self.args.clip_seconds / unit))
		for unit_index in range(num_units):
			start_time = unit_index * unit
			end_time = min((unit_index + 1) * unit, self.args.clip_seconds)
			start_index = int(round(start_time * ctx.video_fps))
			end_index = min(int(round(end_time * ctx.video_fps)), len(clip_poses))
			motion_unit_poses = clip_poses[start_index:end_index]
			
			motion_unit_valid_poses = self._valid_clip_poses(motion_unit_poses)
			motion_unit = self._motion_stats(
				poses=motion_unit_valid_poses,
				duration_s=end_time - start_time,
				prefix="local_"
			)
			relative_first = np.linalg.inv(first_pose) @ motion_unit_valid_poses[0]
			first_yaw, first_pitch, first_roll = self._parse_rotation(relative_first[:3, :3])
			first_position = [self._round(value) for value in relative_first[:3, 3].tolist()]
			motion_units.append({
				"index": unit_index,
				"time_range_s": [self._round(start_time), self._round(end_time)],
				"first_pose_position_m": first_position,
				"first_pose_yaw_deg": self._round(first_yaw),
				"first_pose_pitch_deg": self._round(first_pitch),
				"first_pose_roll_deg": self._round(first_roll),
				**motion_unit,
			})
		write_json(ctx.intermediate_dir / "motion_extraction.json", {
			"overall_motion": trajectory,
			"motion_units": motion_units,
		})


class TextGenerationBackend:
	"""Hugging Face causal LLM backend with deterministic JSON repair.

	The backend keeps generation policy centralized for all language-only stages:
	sampling uses no top-k, `top_p=0.9`, and KV caching; temperature `0.0`
	switches to deterministic decoding. When `cpu_offload` is enabled, the model
	stays on CPU between active sessions and can remain on the worker GPU across
	adjacent stages that share the same backend.
	"""

	def __init__(
		self,
		model_path: str,
		local_rank: int | None,
		cpu_offload: bool = False,
		think_if_available: bool = False,
	):
		self.cpu_offload = cpu_offload
		self.think_if_available = think_if_available
		self._active_lease_count = 0
		self.target_device = torch.device(
			f"cuda:{local_rank}" if torch.cuda.is_available() and local_rank is not None else
			"cuda" if torch.cuda.is_available() else
			"cpu"
		)
		self.tokenizer = AutoTokenizer.from_pretrained(
			model_path,
			local_files_only=True,
			trust_remote_code=True,
		)
		load_kwargs: dict[str, Any] = {
			"local_files_only": True,
			"trust_remote_code": True,
			"torch_dtype": "auto",
		}
		if torch.cuda.is_available() and local_rank is not None and not cpu_offload:
			load_kwargs["device_map"] = {"": local_rank}
		self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
		if "device_map" not in load_kwargs:
			self.model.to("cpu" if cpu_offload else self.target_device)
		self.model.eval()

	def _activate(self) -> torch.device:
		"""Acquire one active-device lease and return the generation device."""

		if self.cpu_offload and self.target_device.type == "cuda" and self._active_lease_count == 0:
			self.model.to(self.target_device)
		self._active_lease_count += 1
		return self.target_device if not self.cpu_offload or self.target_device.type == "cuda" else torch.device("cpu")

	def _deactivate(self) -> None:
		"""Release one active-device lease and offload after the last lease."""

		if self._active_lease_count <= 0:
			return
		self._active_lease_count -= 1
		if self.cpu_offload and self.target_device.type == "cuda" and self._active_lease_count == 0:
			self.model.to("cpu")
			torch.cuda.empty_cache()

	def _generation_kwargs(self, temperature: float, max_new_tokens: int, pad_token_id: int | None = None) -> dict[str, Any]:
		"""Build the shared decoding configuration for text generation."""

		kwargs: dict[str, Any] = {
			"max_new_tokens": max_new_tokens,
			"do_sample": temperature > 0.0,
			"use_cache": True,
		}
		if pad_token_id is not None:
			kwargs["pad_token_id"] = pad_token_id
		if temperature > 0.0:
			kwargs.update({
				"temperature": temperature,
				"top_p": 0.9,
				"top_k": 0,
			})
		return kwargs
	
	def _extract_json_response(self, text: str) -> Any:
		"""Parse the first JSON object from a model response."""

		candidate = text.strip()
		if candidate.startswith("```"):
			lines = candidate.splitlines()
			if lines and lines[0].startswith("```"):
				lines = lines[1:]
			if lines and lines[-1].startswith("```"):
				lines = lines[:-1]
			candidate = "\n".join(lines).strip()
		try:
			return json.loads(candidate)
		except json.JSONDecodeError:
			start = candidate.find("{")
			end = candidate.rfind("}")
			if start >= 0 and end > start:
				return json.loads(candidate[start:(end + 1)])
			raise

	@contextmanager
	def active(self):
		"""Keep an offloaded backend active across a contiguous stage group."""

		try:
			yield self._activate()
		finally:
			self._deactivate()
	
	def generate(self, prompt: str, temperature: float, max_new_tokens: int, media: list[dict[str, Any]] | None = None) -> str:
		"""Generate text from a prompt."""

		del media
		messages = [{"role": "user", "content": prompt}]
		text = self.tokenizer.apply_chat_template(
			messages, tokenize=False, add_generation_prompt=True,
			enable_thinking=self.think_if_available,
		)
		with self.active() as device:
			inputs = self.tokenizer([text], return_tensors="pt").to(device)
			with torch.inference_mode():
				outputs = self.model.generate(
					**inputs,
					**self._generation_kwargs(
						temperature=temperature,
						max_new_tokens=max_new_tokens,
						pad_token_id=self.tokenizer.eos_token_id,
					),
				)
			new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
			return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

	def generate_json(
		self,
		prompt: str,
		temperature: float,
		max_new_tokens: int,
		media: list[dict[str, Any]] | None = None,
	) -> Any:
		"""Generate JSON and perform one deterministic repair if parsing fails."""

		with self.active():
			output = self.generate(prompt, temperature=temperature, max_new_tokens=max_new_tokens, media=media)
			try:
				return self._extract_json_response(output)
			except Exception:
				repair_prompt = (
					"Repair the following model response into valid JSON only. "
					"Do not add Markdown or explanation.\n\n"
					f"Original task:\n{prompt}\n\nInvalid response:\n{output}"
				)
			repaired = self.generate(repair_prompt, temperature=0.0, max_new_tokens=max_new_tokens, media=media)
			try:
				return self._extract_json_response(repaired)
			except Exception as error:
				raise RejectedSample("Model response could not be decoded as JSON after one deterministic repair.") from error
	
	def get_json_key(self, payload: Any, key: str) -> Any:
		"""Return one required generated-JSON value or reject the sample."""

		if not isinstance(payload, dict):
			raise RejectedSample(f"Model JSON output is not an object; expected key `{key}`.")
		if key not in payload:
			available_keys = ", ".join(sorted(str(available_key) for available_key in payload.keys()))
			raise RejectedSample(f"Model JSON output is missing key `{key}`. Available keys: {available_keys or 'none'}.")
		return payload[key]


class VisionLanguageBackend(TextGenerationBackend):
	"""Hugging Face image/video-to-text backend for Qwen3-VL-style models."""

	def __init__(
		self,
		model_path: str,
		local_rank: int | None,
		cpu_offload: bool = False
	):
		self.cpu_offload = cpu_offload
		self._active_lease_count = 0
		self.target_device = torch.device(
			f"cuda:{local_rank}" if torch.cuda.is_available() and local_rank is not None else
			"cuda" if torch.cuda.is_available() else
			"cpu"
		)
		self.processor = AutoProcessor.from_pretrained(
			model_path,
			local_files_only=True,
			trust_remote_code=True,
		)
		load_kwargs: dict[str, Any] = {
			"local_files_only": True,
			"trust_remote_code": True,
			"torch_dtype": "auto",
		}
		if torch.cuda.is_available() and local_rank is not None and not cpu_offload:
			load_kwargs["device_map"] = {"": local_rank}
		self.model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
		if "device_map" not in load_kwargs:
			self.model.to("cpu" if cpu_offload else self.target_device)
		self.model.eval()

	def generate(self, prompt: str, temperature: float, max_new_tokens: int, media: list[dict[str, Any]] | None = None) -> str:
		"""Generate text from optional image/video media plus a prompt."""

		content: list[dict[str, Any]] = []
		for item in media or []:
			if item["type"] == "image":
				content.append({
					"type": "image",
					"image": Path(item["path"]).resolve().as_uri(),
					"resized_width": int(item["resized_width"]),
					"resized_height": int(item["resized_height"]),
				})
			elif item["type"] == "video":
				content.append({
					"type": "video",
					"video": Path(item["path"]).resolve().as_uri(),
					"fps": float(item["fps"]),
					"resized_width": int(item["resized_width"]),
					"resized_height": int(item["resized_height"]),
				})
		content.append({"type": "text", "text": prompt})
		messages = [{"role": "user", "content": content}]
		text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		
		image_inputs, video_inputs, video_kwargs = process_vision_info(
			messages, return_video_kwargs=True, return_video_metadata=True,
			image_patch_size=self.processor.image_processor.patch_size,
		)
		if video_inputs is not None:
			video_inputs, video_metadata = zip(*video_inputs)
			video_inputs, video_metadata = list(video_inputs), list(video_metadata)
		else:
			video_kwargs, video_metadata = {}, None
		
		device = self._activate()
		try:
			inputs = self.processor(
				text=[text],
				images=image_inputs,
				videos=video_inputs,
				video_metadata=video_metadata,
				padding=True,
				return_tensors="pt",
				**video_kwargs,
			).to(device)
			with torch.inference_mode():
				outputs = self.model.generate(
					**inputs,
					**self._generation_kwargs(
						temperature=temperature,
						max_new_tokens=max_new_tokens,
						pad_token_id=self.processor.tokenizer.eos_token_id,
					),
				)
			new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
			return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
		finally:
			self._deactivate()


class MotionDigestingStage:
	"""Digest numeric motion extraction into a natural-language motion caption.

	The input is `motion_extraction.json`, which is intentionally numeric and
	structured. This stage asks the LLM to interpret those numbers into compact
	camera-motion units and one user-readable motion caption while avoiding any
	visual scene claims.
	"""

	def __init__(self, args: Namespace, llm: TextGenerationBackend):
		self.args = args
		self.llm = llm

	def __call__(self, ctx: SampleContext) -> None:
		"""Save `motion_caption.json`."""

		motion_extraction_json = json.dumps(
			read_json(ctx.intermediate_dir / "motion_extraction.json"),
			ensure_ascii=False, indent=2
		)
		prompt = (
			MOTION_DIGESTING_TEMPLATE
			.replace("<MOTION_EXTRACTION_JSON>", motion_extraction_json)
		)
		result = self.llm.generate_json(
			prompt,
			temperature=self.args.motion_digesting_llm_temperature,
			max_new_tokens=self.args.motion_digesting_llm_max_new_tokens,
		)
		write_json(ctx.intermediate_dir / "motion_caption.json", {
			"motion_caption": self.llm.get_json_key(result, "motion_caption"),
		})


class VideoCaptioningStage:
	"""Caption the ground-truth video with VLM visual evidence and motion guidance.

	The VLM sees the square video clip and the motion caption. It is responsible
	for visible objects, layout, and grounding camera motion to scene regions
	without contradicting the pose-derived motion.
	"""

	def __init__(self, args: Namespace, vlm: VisionLanguageBackend):
		self.args = args
		self.vlm = vlm

	def __call__(self, ctx: SampleContext) -> None:
		"""Save `video_caption.json`."""

		motion_caption_json = json.dumps(
			read_json(ctx.intermediate_dir / "motion_caption.json"),
			ensure_ascii=False, indent=2
		)
		prompt = (
			VIDEO_CAPTIONING_TEMPLATE
			.replace("<MOTION_CAPTION_JSON>", motion_caption_json)
		)
		result = self.vlm.generate_json(
			prompt,
			temperature=self.args.video_captioning_vlm_temperature,
			max_new_tokens=self.args.video_captioning_vlm_max_new_tokens,
			media=[{
				"type": "video",
				"path": str(ctx.gt_clip_path),
				"fps": self.args.video_captioning_fps,
				"resized_width": self.args.video_captioning_width,
				"resized_height": self.args.video_captioning_height,
			}],
		)
		write_json(ctx.intermediate_dir / "video_caption.json", {
			"video_caption": self.vlm.get_json_key(result, "video_caption"),
		})


class ImageCaptioningStage:
	"""Caption each ordered reference image independently with a VLM.

	Reference order is already fixed by sampling. Captions therefore explicitly
	name the first, second, and later images so downstream text-only stages can
	safely wire video objects and start-frame references to the correct input
	image indices.
	"""

	def __init__(self, args: Namespace, vlm: VisionLanguageBackend):
		self.args = args
		self.vlm = vlm

	def __call__(self, ctx: SampleContext) -> None:
		"""Save `image_captions.json`."""

		captions: list[str] = []
		for ref in ctx.manifest_entry["ref_imgs"]:
			index = ref["index"]
			is_video_start_clause = "is" if ref["is_start_frame"] else "is not"
			prompt = (
				IMAGE_CAPTIONING_TEMPLATE
				.replace("<REF_INDEX>", str(index))
				.replace("<IS_VIDEO_START_CLAUSE>", is_video_start_clause)
			)
			result = self.vlm.generate_json(
				prompt,
				temperature=self.args.image_captioning_vlm_temperature,
				max_new_tokens=self.args.image_captioning_vlm_max_new_tokens,
				media=[{
					"type": "image",
					"path": str(ctx.ref_img_dir / f"ref_{index:02d}.jpg"),
					"resized_width": self.args.image_captioning_width,
					"resized_height": self.args.image_captioning_height,
				}],
			)
			captions.append(self.vlm.get_json_key(result, "image_caption"))
		write_json(ctx.intermediate_dir / "image_captions.json", {
			"image_captions": captions,
		})


class CaptionWiringStage:
	"""Wire video-caption content to the specific reference-image captions.

	This is the language-only grounding stage. It preserves the video caption as
	the source of truth for the target clip while using reference captions only
	when they support a specific image-index claim.
	"""

	def __init__(self, args: Namespace, llm: TextGenerationBackend):
		self.args = args
		self.llm = llm

	def __call__(self, ctx: SampleContext) -> None:
		"""Save `wired_caption.json`."""

		video_caption_json = json.dumps(
			read_json(ctx.intermediate_dir / "video_caption.json"),
			ensure_ascii=False, indent=2
		)
		image_captions_json = json.dumps(
			read_json(ctx.intermediate_dir / "image_captions.json"),
			ensure_ascii=False, indent=2
		)
		prompt = (
			CAPTION_WIRING_TEMPLATE
			.replace("<VIDEO_CAPTION_JSON>", video_caption_json)
			.replace("<IMAGE_CAPTIONS_JSON>", image_captions_json)
		)
		result = self.llm.generate_json(
			prompt,
			temperature=self.args.caption_wiring_llm_temperature,
			max_new_tokens=self.args.caption_wiring_llm_max_new_tokens,
		)
		write_json(ctx.intermediate_dir / "wired_caption.json", {
			"wired_caption": self.llm.get_json_key(result, "wired_caption"),
		})


class CaptionRephrasingStage:
	"""Rephrase the wired caption as the final instruction-style user prompt.

	The stage changes style, not facts: it turns the descriptive wired caption
	into an imperative request suitable for the video-generation model and stores
	the result on the manifest entry.
	"""

	def __init__(self, args: Namespace, llm: TextGenerationBackend):
		self.args = args
		self.llm = llm

	def __call__(self, ctx: SampleContext) -> None:
		"""Update the manifest entry with `synthesized_prompt`."""

		wired_caption_json = json.dumps(
			read_json(ctx.intermediate_dir / "wired_caption.json"),
			ensure_ascii=False, indent=2
		)
		prompt = (
			CAPTION_REPHRASING_TEMPLATE
			.replace("<WIRED_CAPTION_JSON>", wired_caption_json)
		)
		result = self.llm.generate_json(
			prompt,
			temperature=self.args.caption_rephrasing_llm_temperature,
			max_new_tokens=self.args.caption_rephrasing_llm_max_new_tokens,
		)
		synthesized_prompt = self.llm.get_json_key(result, "synthesized_prompt")
		ctx.manifest_entry["synthesized_prompt"] = synthesized_prompt


class CriticJudgingStage:
	"""Use a deterministic LLM critic as the final validation gate.

	The critic checks the synthesized prompt against motion, video, and reference
	captions. Fatal checks reject immediately; quality checks are averaged and
	compared against `filter_quality_score_min` when that filter is configured.
	"""

	def __init__(self, args: Namespace, llm: TextGenerationBackend):
		self.args = args
		self.llm = llm

	def __call__(self, ctx: SampleContext) -> None:
		"""Save `critic_judging.json` and reject invalid prompt candidates."""

		motion_caption_json = json.dumps(read_json(
			ctx.intermediate_dir / "motion_caption.json"),
			ensure_ascii=False, indent=2
		)
		video_caption_json = json.dumps(read_json(
			ctx.intermediate_dir / "video_caption.json"),
			ensure_ascii=False, indent=2
		)
		image_captions_json = json.dumps(read_json(
			ctx.intermediate_dir / "image_captions.json"),
			ensure_ascii=False, indent=2
		)
		synthesized_prompt = ctx.manifest_entry["synthesized_prompt"]
		prompt = (
			CRITIC_JUDGING_TEMPLATE
			.replace("<MOTION_CAPTION_JSON>", motion_caption_json)
			.replace("<VIDEO_CAPTION_JSON>", video_caption_json)
			.replace("<IMAGE_CAPTIONS_JSON>", image_captions_json)
			.replace("<SYNTHESIZED_PROMPT>", synthesized_prompt)
		)
		result = self.llm.generate_json(
			prompt,
			temperature=self.args.critic_judging_llm_temperature,
			max_new_tokens=self.args.critic_judging_llm_max_new_tokens,
		)
		fatal_checks = self.llm.get_json_key(result, "fatal_checks") or {}
		quality_checks = self.llm.get_json_key(result, "quality_checks") or {}
		fatal_passed = all(bool(value) for value in fatal_checks.values()) and len(fatal_checks) > 0
		quality_values = [bool(value) for value in quality_checks.values()]
		quality_score = sum(quality_values) / len(quality_values) if quality_values else 0.0
		result["computed_quality_score"] = quality_score
		if not fatal_passed:
			raise RejectedSample("LLM judge fatal checks failed.")
		if self.args.filter_quality_score_min is not None and quality_score < self.args.filter_quality_score_min:
			raise RejectedSample(f"LLM judge quality score {quality_score:.4f} below threshold.")
		write_json(ctx.intermediate_dir / "llm_validation.json", result)


class DistillationStage:
	"""Create medium and coarse prompt variants after critic acceptance.

	This final language-only stage broadens prompt granularity while preserving
	the accepted prompt's start view, camera motion, and grounded scene objects.
	The distilled prompts are written into the manifest entry used by training.
	"""

	def __init__(self, args: Namespace, llm: TextGenerationBackend):
		self.args = args
		self.llm = llm

	def __call__(self, ctx: SampleContext) -> None:
		"""Update the manifest entry with distilled prompt variants."""

		synthesized_prompt = ctx.manifest_entry["synthesized_prompt"]
		prompt = (
			DISTILLATION_TEMPLATE
			.replace("<SYNTHESIZED_PROMPT>", synthesized_prompt)
		)
		result = self.llm.generate_json(
			prompt,
			temperature=self.args.distillation_llm_temperature,
			max_new_tokens=self.args.distillation_llm_max_new_tokens,
		)
		ctx.manifest_entry["distilled_prompts"] = {
			"medium": self.llm.get_json_key(result, "medium_prompt"),
			"coarse": self.llm.get_json_key(result, "coarse_prompt"),
		}


def commit_sample(args: Namespace, ctx: SampleContext) -> bool | None:
	"""Atomically publish a sample and update the manifest ledger.

	Returns:
		`True` when committed, `False` for a duplicate, and `None` when the
		global target has already been reached.
	"""

	tmp_dir = ctx.require_tmp_dir()
	final_dir = ctx.require_final_dir()
	with exclusive_lock(args):
		state = read_json(state_path(args))
		if state["current_count"] >= state["target_count"]:
			shutil.rmtree(tmp_dir, ignore_errors=True)
			return None
		if final_dir.exists():
			shutil.rmtree(tmp_dir, ignore_errors=True)
			return False
		write_json(tmp_dir / "sample.json", ctx.manifest_entry)
		os.replace(tmp_dir, final_dir)
		write_jsonl(manifest_path(args), ctx.manifest_entry, append=True)
		state["current_count"] += 1
		write_json(state_path(args), state)
		return True


def prepare_output_root(args: Namespace) -> None:
	"""Prepare output storage and initialize worker state.

	It optionally removes incomplete temporary samples and stale active-split
	manifest entries while preserving samples referenced by split manifest.
	"""

	root = output_root(args)
	root.mkdir(parents=True, exist_ok=True)
	samples_root = root / "samples"
	samples_root.mkdir(parents=True, exist_ok=True)
	manifest = manifest_path(args)
	manifest = manifest_path(args)
	manifest.parent.mkdir(parents=True, exist_ok=True)
	manifest.touch(exist_ok=True)
	
	with exclusive_lock(args):
		if not args.no_cleanup:
			entries = read_jsonl(manifest)
			active_entries: list[dict[str, Any]] = []
			active_sample_ids: set[str] = set()
			
			if not args.restart_fresh:
				for entry in entries:
					sample_id = str(entry.get("sample_id", ""))
					if not sample_id:
						continue
					if sample_id in active_sample_ids:
						continue
					if not (samples_root / sample_id).is_dir():
						continue
					active_sample_ids.add(sample_id)
					active_entries.append(entry)
		
			for child in samples_root.iterdir():
				if not child.is_dir():
					child.unlink(missing_ok=True)
				elif child.name not in active_sample_ids:
					shutil.rmtree(child)
			write_jsonl(manifest, *active_entries, append=False)
		
		write_json(state_path(args), {
			"current_count": len(read_jsonl(manifest)),
			"target_count": args.num_samples,
		})


def cleanup_sample_tmp(ctx: SampleContext) -> None:
	"""Remove a sample temporary directory without risking unrelated paths."""

	tmp_dir = ctx.tmp_dir
	if tmp_dir is None:
		return
	if tmp_dir.name.startswith(".tmp_"):
		shutil.rmtree(tmp_dir, ignore_errors=True)
	ctx.tmp_dir = None


def run_stages(stages: list[Callable[[SampleContext], None]], ctx: SampleContext) -> None:
	"""Run stages while keeping shared offloaded backends resident across groups."""

	def get_backend(stage):
		return getattr(stage, "llm", None) or getattr(stage, "vlm", None)

	index = 0
	while index < len(stages):
		backend = get_backend(stages[index])
		if backend is None:
			stages[index](ctx)
			index += 1
			continue

		group_end = index + 1
		while group_end < len(stages) and get_backend(stages[group_end]) is backend:
			group_end += 1

		with backend.active():
			for stage in stages[index:group_end]:
				stage(ctx)
		index = group_end


def run_worker(args: Namespace, worker_index: int = 0) -> None:
	"""Run one independent sample-generation loop."""

	if torch.cuda.is_available():
		device_count = torch.cuda.device_count()
		if device_count <= 0:
			local_rank = None
		else:
			local_rank = worker_index % device_count
			torch.cuda.set_device(local_rank)
	else:
		local_rank = None

	# Process-local and time-dependent seed
	base = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), "little")
	seed = (base ^ time.time_ns() ^ (os.getpid() << 16) ^ worker_index) & 0xFFFFFFFF

	rng = random.Random(seed)
	np.random.seed(seed)
	state = read_json(state_path(args))
	if state["current_count"] >= state["target_count"]:
		return

	print(f"[worker {worker_index}] loading VLM ...", flush=True)
	vlm = VisionLanguageBackend(
		args.vlm_backend_path,
		local_rank=local_rank,
		cpu_offload=args.vlm_cpu_offload
	)
	print(f"[worker {worker_index}] loading LLM ...", flush=True)
	llm = TextGenerationBackend(
		args.llm_backend_path,
		local_rank=local_rank,
		cpu_offload=args.llm_cpu_offload,
		think_if_available=args.llm_think_if_available,
	)
	stages: list[Callable[[SampleContext], None]] = [
		DataSamplingStage(args, rng),
		MotionExtractionStage(args),
		MotionDigestingStage(args, llm),
		VideoCaptioningStage(args, vlm),
		ImageCaptioningStage(args, vlm),
		CaptionWiringStage(args, llm),
		CaptionRephrasingStage(args, llm),
		CriticJudgingStage(args, llm),
		DistillationStage(args, llm),
	]

	while True:
		state = read_json(state_path(args))
		if state["current_count"] >= state["target_count"]:
			break
		ctx = SampleContext()
		try:
			run_stages(stages, ctx)
			result = commit_sample(args, ctx)
			if result is None:
				break
			if result:
				updated_state = read_json(state_path(args))
				print(f"[worker {worker_index}] committed {ctx.sample_id} ({updated_state['current_count']}/{updated_state['target_count']})", flush=True)
		except RejectedSample as error:
			cleanup_sample_tmp(ctx)
			print(f"[worker {worker_index}] rejected sample: {error}", flush=True)
		except Exception:
			cleanup_sample_tmp(ctx)
			raise


def main() -> None:
	"""Launch one or more independent curation workers."""

	args = parse_args()
	prepare_output_root(args)
	if args.num_processes == 1:
		run_worker(args, worker_index=0)
		return

	context = mp.get_context("spawn")
	processes = [
		context.Process(target=run_worker, args=(args, worker_index))
		for worker_index in range(args.num_processes)
	]
	for process in processes:
		process.start()
	for process in processes:
		process.join()
	failures = [process.exitcode for process in processes if process.exitcode != 0]
	if failures:
		raise SystemExit(f"One or more workers failed with exit codes: {failures}")


if __name__ == "__main__":
	main()
