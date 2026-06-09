from __future__ import annotations

from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
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
import pyiqa
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


class Pose:
	"""Camera-to-world pose matrix with helpers for local camera motion.

	The ScanNet++ iPhone poses used by the clip pipeline follow the OpenCV-like
	camera convention used in the prompts: +X points right, +Y points down, and
	+Z points forward. Relative transforms therefore use
	`inv(start_camera_to_world) @ end_camera_to_world`, which expresses the end
	camera pose in the starting camera's local coordinate system.
	"""

	def __init__(self, array: np.ndarray):
		"""Store one finite 4x4 camera-to-world transform."""

		array = np.asarray(array, dtype=np.float64)
		if array.shape != (4, 4) or not np.isfinite(array).all():
			raise ValueError("Pose expects a finite 4x4 matrix.")
		self.array = array
	
	def __call__(self) -> np.ndarray:
		"""Return the raw 4x4 camera-to-world matrix."""

		return self.array
	
	def serialized(self) -> list[list[float]]:
		"""Return the pose as JSON-serializable nested lists."""

		return self.array.tolist()
	
	@classmethod
	def from_payload(cls, payload: Any) -> Pose | None:
		"""Parse a pose from common ScanNet++ and Nerfstudio payload shapes."""

		if payload is None:
			return None
		if isinstance(payload, dict):
			for field in (
				"transform_matrix",
				"aligned_pose", "pose",
				"camera_to_world", "c2w"
			):
				if field in payload:
					return cls.from_payload(payload[field])
		try:
			array = np.asarray(payload, dtype=np.float64)
		except (TypeError, ValueError):
			return None
		if array.shape == (16,):
			array = array.reshape(4, 4)
		if array.shape != (4, 4) or not np.isfinite(array).all():
			return None
		return cls(array)	
	
	def yz_flipped(self) -> Pose:
		"""Convert an OpenGL-style camera-to-world pose to the OpenCV-style axes."""

		return Pose(self() @ np.diag([1, -1, -1, 1]))

	def relative_to(self, pose: Pose) -> Pose:
		"""Return this pose expressed in another pose's local camera frame."""

		return Pose(np.linalg.inv(pose()) @ self()) 
	
	def translation(self) -> np.ndarray:
		"""Return the camera center in world or relative coordinates."""

		return self()[:3, 3]
	
	def look_direction(self) -> np.ndarray:
		"""Return the normalized +Z camera-forward direction."""

		direction = self()[:3, 2]
		return direction / max(float(np.linalg.norm(direction)), 1e-8)
	
	def rotation(self) -> np.ndarray:
		"""Return the 3x3 rotation block."""

		return self()[:3, :3]
	
	def euler_angle(self) -> tuple[float, float, float]:
		"""Return yaw, pitch, and roll deltas in degrees.

		Positive yaw pans right, positive pitch tilts up, and positive roll rotates
		clockwise in the OpenCV-like camera coordinate frame.
		"""

		R = self.rotation()
		yaw = np.arctan2(R[0, 2], R[2, 2])
		pitch = np.arctan2(-R[1, 2], np.hypot(R[0, 2], R[2, 2]))
		roll = np.arctan2(R[1, 0], R[1, 1])
		return np.degrees([yaw, pitch, roll])
	
	def rotation_angle(self) -> float:
		"""Return the intrinsic rotation angle in degrees."""
		cos_angle = ((np.trace(self.rotation()) - 1.0) / 2.0).clip(-1.0, 1.0)
		return float(np.degrees(np.arccos(cos_angle)))


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
	parser.add_argument("--pose_pool_multiplier", type=int, default=4)
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
	parser.add_argument("--filter_motion_amount_min", type=float, default=None)
	parser.add_argument("--filter_motion_amount_max", type=float, default=None)
	parser.add_argument("--filter_motion_unsteadiness_max", type=float, default=None)
	parser.add_argument("--filter_dslr_brisque_score_max", type=float, default=None)
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
	if args.pose_pool_multiplier < 1:
		raise ValueError("`--pose_pool_multiplier` must be at least 1.")
	if not 0.0 <= args.include_start_frame_prob <= 1.0:
		raise ValueError("`--include_start_frame_prob` must be in [0, 1].")
	if args.motion_digesting_unit_seconds <= 0:
		raise ValueError("`--motion_digesting_unit_seconds` must be positive.")
	if args.video_captioning_fps <= 0:
		raise ValueError("`--video_captioning_fps` must be positive.")
	for name in (
		"filter_pixel_valid_fraction_min",
		"filter_pose_valid_fraction_min",
		"filter_motion_amount_min",
		"filter_motion_amount_max",
		"filter_motion_unsteadiness_max",
		"filter_dslr_brisque_score_max",
		"filter_quality_score_min",
	):
		value = getattr(args, name)
		if value is not None and value < 0:
			raise ValueError(f"`--{name}` must be non-negative when provided.")
	for name in (
		"filter_pixel_valid_fraction_min",
		"filter_pose_valid_fraction_min",
		"filter_quality_score_min"
	):
		value = getattr(args, name)
		if value is not None and value > 1:
			raise ValueError(f"`--{name}` must be no greater than 1 when provided.")
	if (
		args.filter_motion_amount_min is not None and
		args.filter_motion_amount_max is not None and
		args.filter_motion_amount_min > args.filter_motion_amount_max
	):
		raise ValueError("`--filter_motion_amount_min` cannot exceed `--filter_motion_amount_max`.")
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
	*payload: dict[str, Any],
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
	and writes the sample-local media files under a `samples/.tmp_*` directory.
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
		self._brisque_metric: Any | None = None
	
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
				return max(int(value), 0)
			except ValueError:
				return 0
		
		source_video_path = ctx.require_source_video_path()
		if not source_video_path.exists():
			raise RejectedSample(f"Missing iPhone video: {source_video_path}")
		probe = ffmpeg.probe(str(source_video_path), select_streams="v:0")
		streams = probe.get("streams") or []
		if not streams:
			raise RejectedSample(f"No video stream found in {source_video_path}")
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
			raise RejectedSample(f"Could not determine FPS for video: {source_video_path}")
			
		num_frames = (
			parse_num_frames(stream.get("nb_frames"))
			or int(round(frame_rate * (duration or 0.0)))
		)
		if num_frames <= 0:
			raise RejectedSample(f"Could not determine frame count for video: {source_video_path}")

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
	
	def _prepare_poses(self, ctx: SampleContext) -> None:
		"""Store the sampled iPhone pose slice as clean JSON matrices."""

		def sort_key(key: Any) -> tuple[int, str]:
			text = str(key)
			if text.isdigit():
				return int(text), text
			stem = Path(text).stem
			digits = "".join(character for character in stem if character.isdigit())
			if digits:
				return int(digits), text
			return float(10**12), text

		def parse_payload_to_sequence(payload: Any) -> list[Any]:
			if payload is None:
				return []
			if isinstance(payload, dict):
				for field in ("aligned_poses", "poses", "frames"):
					if field in payload:
						return parse_payload_to_sequence(payload[field])
				items = sorted(payload.items(), key=lambda item: sort_key(item[0]))
				return [value for _, value in items]
			return list(payload)

		path = ctx.require_source_pose_path()
		if not path.exists():
			raise RejectedSample(f"Missing iPhone pose file: {path}")
		raw_payload = read_json(path)
		payload_sequence = parse_payload_to_sequence(raw_payload)
		if not payload_sequence:
			raise RejectedSample(f"No pose records found in {path}")
		poses = [Pose.from_payload(p) for p in payload_sequence]
		clip_poses = poses[ctx.start_frame:(ctx.start_frame + ctx.clip_frames)]
		if len(clip_poses) < ctx.clip_frames:
			raise RejectedSample("Pose sequence is shorter than the sampled clip.")
		write_json(ctx.intermediate_dir / "poses.json", {"poses":
			[p.serialized() if p is not None else None for p in clip_poses]
		})

	def _ref_image_brisque_score(self, image: Image.Image) -> float:
		"""Return the BRISQUE no-reference quality score for one RGB image."""
		
		array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
		tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
		with torch.inference_mode():
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			metric = self._brisque_metric
			if metric is None:
				metric = pyiqa.create_metric("brisque", device=device)
				self._brisque_metric = metric
			score = metric(tensor.to(device)).item()
		return float(score)

	def _ref_image_pose_relevance_score(
		self, dslr_pose: Pose | None, clip_poses: list[Pose | None],
	) -> float:
		"""Return the best DSLR-to-clip pose relevance score in [0, 1]."""
	
		best_score = 0.0
		if dslr_pose is None:
			return best_score
		dslr_pose = dslr_pose.yz_flipped()
		depth, m_sigma, deg_sigma = 3.0, 1.0, 45.0
		dslr_pos = dslr_pose.translation()
		dslr_look_direction = dslr_pose.look_direction()
		dslr_focus = dslr_pos + dslr_look_direction * depth
		for clip_pose in [p for p in clip_poses if p is not None]:
			clip_pos = clip_pose.translation()
			clip_look_direction = clip_pose.look_direction()
			clip_focus = clip_pos + clip_look_direction * depth
			focus_distance = np.linalg.norm(clip_focus - dslr_focus)
			distance_score = np.exp(-0.5 * (focus_distance / m_sigma) ** 2)
			cos_look_angle = np.dot(clip_look_direction, dslr_look_direction)
			cos_look_angle = cos_look_angle.clip(-1.0, 1.0)
			look_angle = np.degrees(np.arccos(cos_look_angle))
			angle_score = np.exp(-0.5 * (look_angle / deg_sigma) ** 2)
			score = distance_score * angle_score
			best_score = max(best_score, float(score))
		return best_score

	def _prepare_reference_images(
		self, ctx: SampleContext,
		scene: ScannetppScene_Release
	) -> None:
		"""Select relevant references and copy them into fixed sample order."""

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

		def valid_fraction(mask: Image.Image) -> float:
			mask = np.asarray(mask)
			if mask.ndim == 3:
				mask = np.max(mask[..., :3], axis=-1)
			return float((mask > 127).mean())

		def resolve_dslr_path(value: str | None, roots: list[Path]) -> Path | None:
			if not value:
				return None
			file_path = Path(value)
			candidates = [file_path] + [root / file_path for root in roots]
			return next((path for path in candidates if path.exists()), None)

		include_start = self.rng.random() < self.args.include_start_frame_prob
		dslr_path = scene.dslr_nerfstudio_transform_undistorted_path
		if not dslr_path.exists():
			raise RejectedSample(f"Missing DSLR transforms: {dslr_path}")
		payload = read_json(dslr_path)
		dslr_frames = list(payload.get("frames", [])) + list(payload.get("test_frames", []))
		ref_needed = self.rng.randint(1, self.args.max_ref_images)
		dslr_needed = ref_needed - int(include_start)
		ref_sources: list[dict[str, Any]] = []
		
		if include_start:
			mask_path = ctx.require_source_video_mask_path()
			if mask_path.exists() and self.args.filter_pixel_valid_fraction_min is not None:
				start_mask = im_read(mask_path, ctx.start_frame)
				fraction = valid_fraction(square_gray(start_mask))
				if fraction < self.args.filter_pixel_valid_fraction_min:
					raise RejectedSample(f"Reference valid fraction {fraction:.4f} below threshold.")
			image = square_rgb(im_read(ctx.gt_clip_path, 0))
			ref_sources.append({
				"id": f"iphone:{ctx.start_frame:07d}",
				"image": image,
			})
		
		raw_payload = read_json(ctx.intermediate_dir / "poses.json")
		payload_sequence = raw_payload.get("poses", [])
		clip_poses = [Pose.from_payload(p) for p in payload_sequence]
		dslr_candidates: list[tuple[float, int]] = []
		for index, frame in enumerate(dslr_frames):
			dslr_pose = Pose.from_payload(frame)
			score = self._ref_image_pose_relevance_score(dslr_pose, clip_poses)
			dslr_candidates.append((score, index))
		dslr_pooled = dslr_needed * self.args.pose_pool_multiplier
		dslr_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
		dslr_candidates = dslr_candidates[:dslr_pooled]
		self.rng.shuffle(dslr_candidates)
		
		dslr_sources: list[dict[str, Any]] = []
		dslr_count = 0
		while dslr_candidates and dslr_count < dslr_needed:
			_, index = dslr_candidates.pop()
			frame = dslr_frames[index]
			image_path = resolve_dslr_path(
				value=frame.get("file_path"),
				roots=[scene.dslr_resized_undistorted_dir]
			)
			if not image_path or frame.get("is_bad"):
				continue
			mask_path = resolve_dslr_path(
				value=frame.get("mask_path"),
				roots=[scene.dslr_resized_undistorted_mask_dir]
			)
			if mask_path and self.args.filter_pixel_valid_fraction_min is not None:
				with Image.open(mask_path) as loaded:
					fraction = valid_fraction(square_gray(loaded))
				if fraction < self.args.filter_pixel_valid_fraction_min:
					continue
			with Image.open(image_path) as loaded:
				image = square_rgb(loaded.copy())
			if self.args.filter_dslr_brisque_score_max is not None:
				score = self._ref_image_brisque_score(image)
				if score > self.args.filter_dslr_brisque_score_max:
					continue
			dslr_count += 1
			dslr_sources.append({
				"id": f"dslr:{image_path.name}",
				"image": image,
			})
		if len(dslr_sources) < dslr_needed:
			raise RejectedSample(f"Scene {ctx.scene_id} DSLR images not enough after relevance and quality filtering.")
		ref_sources.extend(dslr_sources)
		self.rng.shuffle(ref_sources)

		ref_dir = ctx.ref_img_dir
		ref_dir.mkdir(parents=True, exist_ok=True)
		metadata: list[dict[str, Any]] = []
		for index, source in enumerate(ref_sources, start=1):
			output_name = f"ref_{index:03d}.jpg"
			image: Image.Image = source["image"]
			image.save(ref_dir / output_name, quality=95)
			metadata.append({
				"index": index,
				"width": image.width,
				"height": image.height,
				"path": f"samples/{ctx.sample_id}/ref_imgs/{output_name}",
				"is_start_frame": source["id"].startswith("iphone"),
			})
		ctx.manifest_entry["ref_imgs"] = metadata

	def __call__(self, ctx: SampleContext) -> None:
		"""Create one sampled candidate under a temporary sample directory."""
		
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
		# ctx.sample_id = f"{ctx.scene_id}__f{ctx.start_frame:07d}_a{os.getpgid()}"
		ctx.sample_id = f"{self.rng.getrandbits(32):08x}"
		ctx.tmp_dir = self.samples_root / f".tmp_{ctx.sample_id}"
		ctx.final_dir = self.samples_root / ctx.sample_id
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

	def _valid_clip_poses(self, poses: list[Pose | None]) -> list[Pose]:
		"""Return valid poses and enforce the configured valid-pose fraction."""
		
		valid_poses = [pose for pose in poses if pose is not None]
		if len(valid_poses) < 2:
			raise RejectedSample("Fewer than two valid poses in sampled clip.")
		valid_fraction = len(valid_poses) / len(poses)
		if self.args.filter_pose_valid_fraction_min is not None and valid_fraction < self.args.filter_pose_valid_fraction_min:
			raise RejectedSample(f"Pose valid fraction {valid_fraction:.4f} below threshold.")
		return valid_poses
		
	def _path_stats(self, poses: list[Pose]) -> dict[str, float]:
		"""Compute translation, rotation, and steadiness statistics for poses."""

		def normalized_std(values: list[float]) -> float:
			if len(values) < 2:
				return 0.0
			mean = np.mean(values)
			if mean <= 1e-8:
				return 0.0
			return float(np.std(values) / mean)

		distances: list[float] = []
		rot_angles: list[float] = []
		trans_angles: list[float] = []
		for i in range(1, len(poses)):
			rel_pose = poses[i].relative_to(poses[i - 1])
			rot_angles.append(rel_pose.rotation_angle())
			translation = rel_pose.translation()
			distance = float(np.linalg.norm(translation))
			distances.append(distance)
			if distance > 1e-8:
				trans_angle = np.degrees(np.arccos(translation[2] / distance))
				trans_angles.append(float(trans_angle))

		path_length = sum(distances)
		motion_amount = path_length + sum(rot_angles) / 90.0
		motion_unsteadiness = (
			normalized_std(distances)
			+ normalized_std(rot_angles)
			+ normalized_std(trans_angles)
		) / 3.0
		return {
			"path_length": path_length,
			"motion_amount": motion_amount,
			"motion_unsteadiness": motion_unsteadiness,
		}
	
	def _relative_stats(self, start: Pose, end: Pose) -> dict[str, float]:
		"""Compute net local translation and signed yaw/pitch/roll from start to end."""

		rel_pose = end.relative_to(start)
		right, down, forward = [float(v) for v in rel_pose.translation()]
		yaw, pitch, roll = [float(v) for v in rel_pose.euler_angle()]
		return {
			"chord_length": float(np.linalg.norm(rel_pose.translation())),
			"right": right, "down": down, "forward": forward,
			"yaw": yaw, "pitch": pitch, "roll": roll,
		}

	def __call__(self, ctx: SampleContext) -> None:
		"""Compute and save `motion_extraction.json`."""

		def round_stats(stats: dict[str, float]) -> dict[str, float]:
			return {k: round(v, 1) for k, v in stats.items()}

		raw_payload = read_json(ctx.intermediate_dir / "poses.json")
		payload_sequence = raw_payload.get("poses", [])
		raw_clip_poses = [Pose.from_payload(p) for p in payload_sequence]
		clip_poses = self._valid_clip_poses(raw_clip_poses)
		path_stats = self._path_stats(clip_poses)
		clip_stats = self._relative_stats(clip_poses[0], clip_poses[-1])
		
		if self.args.filter_motion_amount_min is not None and path_stats["motion_amount"] < self.args.filter_motion_amount_min:
			raise RejectedSample(f"Motion amount {path_stats['motion_amount']:.4f} below threshold.")
		if self.args.filter_motion_amount_max is not None and path_stats["motion_amount"] > self.args.filter_motion_amount_max:
			raise RejectedSample(f"Motion amount {path_stats['motion_amount']:.4f} above threshold.")
		if self.args.filter_motion_unsteadiness_max is not None and path_stats["motion_unsteadiness"] > self.args.filter_motion_unsteadiness_max:
			raise RejectedSample(f"Motion unsteadiness {path_stats['motion_unsteadiness']:.4f} above threshold.")
		
		overall_motion = round_stats({
			"clip_duration_(s)": self.args.clip_seconds,
			"clip_start_to_clip_end_path_length_(m)": path_stats["path_length"],
			"clip_start_to_clip_end_chord_length_(m)": clip_stats["chord_length"],
			"clip_end_in_clip_start_right_(m)": clip_stats["right"],
			"clip_end_in_clip_start_down_(m)": clip_stats["down"],
			"clip_end_in_clip_start_forward_(m)": clip_stats["forward"],
			"clip_end_in_clip_start_yaw_(deg)": clip_stats["yaw"],
			"clip_end_in_clip_start_pitch_(deg)": clip_stats["pitch"],
			"clip_end_in_clip_start_roll_(deg)": clip_stats["roll"],
		})
		
		motion_units: list[dict[str, Any]] = []
		unit_secs = self.args.motion_digesting_unit_seconds
		num_units = int(math.ceil(self.args.clip_seconds / unit_secs))
		for unit_index in range(num_units):
			start_time = unit_index * unit_secs
			end_time = min((unit_index + 1) * unit_secs, self.args.clip_seconds)
			start_index = int(round(start_time * ctx.video_fps))
			end_index = min(int(round(end_time * ctx.video_fps)), len(raw_clip_poses))
			raw_unit_poses = raw_clip_poses[start_index:end_index]
			unit_poses = self._valid_clip_poses(raw_unit_poses)
			anchor_stats = self._relative_stats(clip_poses[0], unit_poses[0])
			path_stats = self._path_stats(unit_poses)
			unit_stats = self._relative_stats(unit_poses[0], unit_poses[-1])
			
			motion_units.append(round_stats({
				"unit_index": unit_index,
				"unit_begins_at_clip_(s)": start_time,
				"unit_duration_(s)": end_time - start_time,
				"unit_start_in_clip_start_right_(m)": anchor_stats["right"],
				"unit_start_in_clip_start_down_(m)": anchor_stats["down"],
				"unit_start_in_clip_start_forward_(m)": anchor_stats["forward"],
				"unit_start_in_clip_start_yaw_(deg)": anchor_stats["yaw"],
				"unit_start_in_clip_start_pitch_(deg)": anchor_stats["pitch"],
				"unit_start_in_clip_start_roll_(deg)": anchor_stats["roll"],
				"unit_start_to_unit_end_path_length_(m)": path_stats["path_length"],
				"unit_start_to_unit_end_chord_length_(m)": unit_stats["chord_length"],
				"unit_end_in_unit_start_right_(m)": unit_stats["right"],
				"unit_end_in_unit_start_down_(m)": unit_stats["down"],
				"unit_end_in_unit_start_forward_(m)": unit_stats["forward"],
				"unit_end_in_unit_start_yaw_(deg)": unit_stats["yaw"],
				"unit_end_in_unit_start_pitch_(deg)": unit_stats["pitch"],
				"unit_end_in_unit_start_roll_(deg)": unit_stats["roll"],
			}))
		write_json(ctx.intermediate_dir / "motion_extraction.json", {
			"overall_motion": overall_motion,
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
	
	def json_to_str(self, value: Any) -> str:
		return json.dumps(value, ensure_ascii=False, indent=2)
	
	def json_by_key(self, payload: Any, key: str) -> Any:
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

		clip_seconds = f"{ctx.clip_duration_s:g}"
		unit_seconds = f"{self.args.motion_digesting_unit_seconds:g}"
		path = ctx.intermediate_dir / "motion_extraction.json"
		motion_extraction_json = self.llm.json_to_str(read_json(path))
		prompt = (
			MOTION_DIGESTING_TEMPLATE
			.replace("<CLIP_SECONDS>", clip_seconds)
			.replace("<UNIT_SECONDS>", unit_seconds)
			.replace("<MOTION_EXTRACTION_JSON>", motion_extraction_json)
		)
		result = self.llm.generate_json(
			prompt,
			temperature=self.args.motion_digesting_llm_temperature,
			max_new_tokens=self.args.motion_digesting_llm_max_new_tokens,
		)
		write_json(ctx.intermediate_dir / "motion_caption.json", {
			"motion_caption": self.llm.json_by_key(result, "motion_caption"),
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

	def _get_timeline(self, ctx: SampleContext) -> list[dict[str, Any]]:
		"""Describe the frame times implied by the configured VLM video sampling rate."""

		fps = self.args.video_captioning_fps
		duration = ctx.clip_duration_s
		frame_count = int(math.floor( duration* fps)) + 1
		return [{
			"index": i + 1,
			"frame_timestamp_(s)": round(min(i / fps, duration), 1),
		} for i in range(frame_count)]
		

	def __call__(self, ctx: SampleContext) -> None:
		"""Save `video_caption.json`."""

		clip_seconds = f"{ctx.clip_duration_s:g}"
		path = ctx.intermediate_dir / "motion_caption.json"
		motion_caption_json = self.vlm.json_to_str(read_json(path))
		frame_timeline_json = self.vlm.json_to_str(self._get_timeline(ctx))
		prompt = (
			VIDEO_CAPTIONING_TEMPLATE
			.replace("<CLIP_SECONDS>", clip_seconds)
			.replace("<MOTION_CAPTION_JSON>", motion_caption_json)
			.replace("<VIDEO_FRAME_TIMELINE_JSON>", frame_timeline_json)
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
			"video_caption": self.vlm.json_by_key(result, "video_caption"),
		})


class ImageCaptioningStage:
	"""Caption each ordered reference image independently with a VLM.

	Reference order and start-frame flags are fixed by sampling and stored
	separately from each VLM caption so the wiring stage can use those facts
	without making each still-image caption mention hidden metadata.
	"""

	def __init__(self, args: Namespace, vlm: VisionLanguageBackend):
		self.args = args
		self.vlm = vlm

	def __call__(self, ctx: SampleContext) -> None:
		"""Save `image_captions.json`."""
		captions: list[dict[str, Any]] = []
		for ref in ctx.manifest_entry["ref_imgs"]:
			result = self.vlm.generate_json(
				IMAGE_CAPTIONING_TEMPLATE,
				temperature=self.args.image_captioning_vlm_temperature,
				max_new_tokens=self.args.image_captioning_vlm_max_new_tokens,
				media=[{
					"type": "image",
					"path": str(ctx.ref_img_dir / f"ref_{ref["index"]:03d}.jpg"),
					"resized_width": self.args.image_captioning_width,
					"resized_height": self.args.image_captioning_height,
				}],
			)
			captions.append({
				"reference_index": ref["index"],
				"is_video_start_frame": bool(ref["is_start_frame"]),
				"caption": self.vlm.json_by_key(result, "image_caption"),
			})
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

		path = ctx.intermediate_dir / "video_caption.json"
		video_caption_json = self.llm.json_to_str(read_json(path))
		path = ctx.intermediate_dir / "image_captions.json"
		image_captions_json = self.llm.json_to_str(read_json(path))
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
			"wired_caption": self.llm.json_by_key(result, "wired_caption"),
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

		clip_seconds = f"{ctx.clip_duration_s:g}"
		path = ctx.intermediate_dir / "wired_caption.json"
		wired_caption_json = self.llm.json_to_str(read_json(path))
		prompt = (
			CAPTION_REPHRASING_TEMPLATE
			.replace("<CLIP_SECONDS>", clip_seconds)
			.replace("<WIRED_CAPTION_JSON>", wired_caption_json)
		)
		result = self.llm.generate_json(
			prompt,
			temperature=self.args.caption_rephrasing_llm_temperature,
			max_new_tokens=self.args.caption_rephrasing_llm_max_new_tokens,
		)
		synthesized_prompt = self.llm.json_by_key(result, "synthesized_prompt")
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
		"""Save `llm_validation.json` and reject invalid prompt candidates."""

		path = ctx.intermediate_dir / "motion_caption.json"
		motion_caption_json = self.llm.json_to_str(read_json(path))
		path = ctx.intermediate_dir / "video_caption.json"
		video_caption_json = self.llm.json_to_str(read_json(path))
		path = ctx.intermediate_dir / "image_captions.json"
		image_captions_json = self.llm.json_to_str(read_json(path))
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
		fatal_checks = self.llm.json_by_key(result, "fatal_checks") or {}
		quality_checks = self.llm.json_by_key(result, "quality_checks") or {}
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
			"medium": self.llm.json_by_key(result, "medium_prompt"),
			"coarse": self.llm.json_by_key(result, "coarse_prompt"),
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


def run_stages(
	stages: list[Callable[[SampleContext], None]],
	ctx: SampleContext
) -> None:
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
