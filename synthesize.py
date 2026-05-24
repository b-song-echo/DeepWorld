# TODO: Change this to `from argparse import Namespace, Namespace`, so that the type anootations become shorter.
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import imageio.v3 as iio
import numpy as np
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info

from scannetpp.common.scene_release import ScannetppScene_Release
from src.prompts import *


# TODO: For function whose role clealy belongs to a certain stage and is only called inside that stage, put it inside that class as a method. It makes this file more readable and clearer in intent. But note that being called once doesn't necessarily means the function belongs to a stage, it is more about semantic-level functionality grouping.


class RejectedSample(Exception):
	"""Expected rejection for a sampled candidate that fails a curation gate."""


@dataclass
class VideoInfo:
	"""Basic video stream metadata used for deterministic clip sampling."""

	width: int
	height: int
	fps: float
	num_frames: int
	duration_s: float


@dataclass
class DslrCandidate:
	"""One valid DSLR reference-image candidate."""

	image_path: Path
	mask_path: Path | None
	ref_id: str


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

	scene_id: str
	scene_type: str
	scene: ScannetppScene_Release
	sample_id: str
	start_frame: int
	clip_frames: int
	video_fps: float
	tmp_dir: Path
	final_dir: Path
	manifest_entry: dict[str, Any]

	@property
	def intermediate_dir(self) -> Path:
		"""Return the intermediate-output directory for this sample."""

		return self.tmp_dir / "intermediate"


def parse_args() -> argparse.Namespace:
	"""Parse the offline command line."""

	parser = argparse.ArgumentParser(description="Generate curated DeepWorld samples from ScanNet++ scenes.")
	parser.add_argument("--scannetpp_root", type=str, required=True)
	parser.add_argument("--output_root", type=str, required=True)
	parser.add_argument("--num_processes", type=int, default=1)
	parser.add_argument("--split", type=str, choices=["train", "val"], required=True)
	parser.add_argument("--num_samples", type=int, required=True)
	parser.add_argument("--clip_seconds", type=float, default=5.0)
	parser.add_argument("--max_ref_images", type=int, default=8)
	parser.add_argument("--include_start_frame_prob", type=float, default=0.4)
	parser.add_argument("--vlm_backend_path", type=str, required=True)
	parser.add_argument("--vlm_captioning_video_width", type=int, default=720)
	parser.add_argument("--vlm_captioning_video_height", type=int, default=720)
	parser.add_argument("--vlm_captioning_video_fps", type=float, default=2.0)
	parser.add_argument("--vlm_captioning_image_width", type=int, default=720)
	parser.add_argument("--vlm_captioning_image_height", type=int, default=720)
	parser.add_argument("--vlm_captioning_temperature", type=float, default=0.2)
	parser.add_argument("--vlm_captioning_max_new_tokens", type=int, default=2048)
	parser.add_argument("--llm_backend_path", type=str, required=True)
	parser.add_argument("--llm_trajectory_digest_unit_seconds", type=float, default=1.0)
	parser.add_argument("--llm_trajectory_digest_temperature", type=float, default=0.2)
	parser.add_argument("--llm_rewriting_temperature", type=float, default=0.4)
	parser.add_argument("--llm_rewriting_max_new_tokens", type=int, default=4096)
	parser.add_argument("--filter_pixel_valid_fraction_min", type=float, default=None)
	parser.add_argument("--filter_pose_valid_fraction_min", type=float, default=None)
	parser.add_argument("--filter_camera_trajectory_length_m_min", type=float, default=None)
	parser.add_argument("--filter_camera_trajectory_length_m_max", type=float, default=None)
	parser.add_argument("--filter_llm_judge_quality_score", type=float, default=None)
	parser.add_argument("--json_retries", type=int, default=2)
	parser.add_argument("--seed", type=int, default=None)
	args = parser.parse_args()
	
	if args.num_processes < 1:
		raise ValueError(f"`--num_processes` must be positive, got {args.num_processes}.")
	if args.num_samples < 0:
		raise ValueError(f"`--num_samples` must be non-negative, got {args.num_samples}.")
	if args.clip_seconds <= 0:
		raise ValueError(f"`--clip_seconds` must be positive, got {args.clip_seconds}.")
	if args.max_ref_images < 1:
		raise ValueError(f"`--max_ref_images` must be at least 1, got {args.max_ref_images}.")
	if not 0.0 <= args.include_start_frame_prob <= 1.0:
		raise ValueError("`--include_start_frame_prob` must be in [0, 1].")
	if args.llm_trajectory_digest_unit_seconds <= 0:
		raise ValueError("`--llm_trajectory_digest_unit_seconds` must be positive.")
	return args


def output_root(args: argparse.Namespace) -> Path:
	"""Return the output root path."""

	return Path(args.output_root)


def manifest_path(args: argparse.Namespace) -> Path:
	"""Return the split-specific manifest path."""

	return output_root(args) / f"manifest_{args.split}.jsonl"


def state_path(args: argparse.Namespace) -> Path:
	"""Return the split-specific shared state path."""

	return output_root(args) / f"state_{args.split}.json"


def lock_path(args: argparse.Namespace) -> Path:
	"""Return the global runtime lock path."""

	return output_root(args) / "runtime_lock.lock"


# TODO: For cross-process access, use FileLock from filelock. It is more suitable for my case because I need to acess two files (the state and the manifest) simultaneously by locking/unlocking a separate lock file rather than the target files themselves. When the lock is on, the process has exclusive access to the state JSON file and the manifest JSONL file.
@contextlib.contextmanager
def exclusive_lock(path: Path):
	"""Hold an advisory exclusive lock for cross-process manifest updates."""

	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a+", encoding="utf-8") as handle:
		fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
		try:
			yield
		finally:
			fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_json(path: Path) -> Any:
	"""Read one JSON file."""

	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
	"""Write one formatted JSON file."""

	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, ensure_ascii=False)
		handle.write("\n")


# TODO: There is really no need for this stomic writing mechanism, because both the state JSON file is always accessed when then lock is on, so any writing is automatically atomic here. Delete this function.
def write_json_atomic(path: Path, payload: Any) -> None:
	"""Atomically write one JSON file next to the final path."""

	tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
	write_json(tmp_path, payload)
	os.replace(tmp_path, path)


def count_manifest(path: Path) -> int:
	"""Count committed non-empty JSONL entries."""

	if not path.exists():
		return 0
	with path.open("r", encoding="utf-8") as handle:
		return sum(1 for line in handle if line.strip())


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
	"""Append one manifest JSONL entry and fsync it."""

	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8", buffering=1) as handle:
		handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
		handle.flush()
		os.fsync(handle.fileno())


def initialize_state(args: argparse.Namespace) -> None:
	"""Initialize split state from the manifest ledger under the runtime lock."""

	root = output_root(args)
	(root / "samples").mkdir(parents=True, exist_ok=True)
	manifest = manifest_path(args)
	manifest.parent.mkdir(parents=True, exist_ok=True)
	manifest.touch(exist_ok=True)
	with exclusive_lock(lock_path(args)):
		current_count = count_manifest(manifest)
		write_json_atomic(state_path(args), {
			"current_count": current_count,
			"target_count": args.num_samples,
		})


# TODO: There is no need for this separate function. Delete it.
def read_state(args: argparse.Namespace) -> dict[str, int]:
	"""Read split state from disk."""

	return read_json(state_path(args))


# TODO: This function is not consistent with the next function `load_scene_ids`, which takes in args directly. Why do you have to do such wierd decisions? You must carefully inspect the entire codebase to make API more consistent in terms of naming conventions, usage, functionalities, etc.
def load_scene_types(scannetpp_root: Path) -> dict[str, str]:
	"""Load scene-type metadata if it is available."""

	path = scannetpp_root / "metadata" / "scene_types.json"
	if not path.exists():
		return {}
	return read_json(path)


def load_scene_ids(args: argparse.Namespace) -> list[str]:
	"""Load the requested ScanNet++ split scene IDs."""

	split_name = "nvs_sem_train.txt" if args.split == "train" else "nvs_sem_val.txt"
	path = Path(args.scannetpp_root) / "split" / split_name
	if not path.exists():
		raise FileNotFoundError(f"ScanNet++ split file not found: {path}")
	with path.open("r", encoding="utf-8") as handle:
		scene_ids = [line.strip() for line in handle if line.strip()]
	if len(scene_ids) == 0:
		raise ValueError(f"ScanNet++ split file is empty: {path}")
	return scene_ids


def rational_to_float(value: str | None) -> float | None:
	"""Parse an FFprobe rational number such as `60000/1001`."""

	if value is None or value in {"0/0", "N/A", ""}:
		return None
	if "/" in value:
		numerator, denominator = value.split("/", 1)
		den = float(denominator)
		if den == 0:
			return None
		return float(numerator) / den
	return float(value)


# TODO: For anything that involves ffmpeg, use ffmpeg-python rather than directly run the CLI in a subprocess.
def probe_video(path: Path) -> VideoInfo:
	"""Probe video stream metadata with FFprobe, falling back to imageio."""

	# TODO: use ffmpeg.probe for extracting metadata.
	cmd = [
		"ffprobe", "-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,duration",
		"-of", "json",
		str(path),
	]
	try:
		result = subprocess.run(cmd, check=True, capture_output=True, text=True)
		streams = json.loads(result.stdout).get("streams") or []
		if streams:
			stream = streams[0]
			fps = rational_to_float(stream.get("avg_frame_rate")) or rational_to_float(stream.get("r_frame_rate"))
			duration = float(stream.get("duration") or 0.0)
			num_frames_raw = stream.get("nb_frames")
			num_frames = int(num_frames_raw) if num_frames_raw and str(num_frames_raw).isdigit() else 0
			if fps and num_frames <= 0 and duration > 0:
				num_frames = int(round(duration * fps))
			if fps and num_frames > 0:
				return VideoInfo(
					width=int(stream["width"]),
					height=int(stream["height"]),
					fps=float(fps),
					num_frames=num_frames,
					duration_s=duration if duration > 0 else num_frames / float(fps),
				)
	except Exception:
		pass

	# TODO: There is no need for this fallback. FFprobe is installed on the target machine.
	meta = iio.immeta(path)
	props = iio.improps(path)
	fps = float(meta.get("fps") or 0.0)
	if fps <= 0:
		raise RuntimeError(f"Could not determine FPS for video: {path}")
	shape = props.shape
	if len(shape) < 4:
		raise RuntimeError(f"Could not determine video shape for: {path}")
	num_frames, height, width = int(shape[0]), int(shape[1]), int(shape[2])
	return VideoInfo(width=width, height=height, fps=fps, num_frames=num_frames, duration_s=num_frames / fps)


# TODO: Avoid running CLI in a subprocess, instead, use python tools. Delete this function.
def run_checked(cmd: list[str]) -> None:
	"""Run a subprocess and raise a compact error on failure."""

	result = subprocess.run(cmd, capture_output=True, text=True)
	if result.returncode != 0:
		message = result.stderr.strip() or result.stdout.strip()
		raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{message}")


# TODO: Use python ffmpeg API, rather than directly calling CLI in a subprocess.
def ffmpeg_crop_expr() -> str:
	"""Return a centered-square FFmpeg crop expression."""

	return "crop=min(iw\\,ih):min(iw\\,ih):(iw-min(iw\\,ih))/2:(ih-min(iw\\,ih))/2"


def extract_square_clip(
	video_path: Path,
	output_path: Path,
	start_frame: int,
	fps: float,
	duration_s: float,
) -> None:
	"""Extract and center-crop one clip while preserving the source FPS."""

	# TODO: Use python ffmpeg API, rather than directly calling CLI in a subprocess.
	start_time = start_frame / fps
	cmd = [
		"ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
		"-ss", f"{start_time:.9f}",
		"-i", str(video_path),
		"-t", f"{duration_s:.9f}",
		"-vf", ffmpeg_crop_expr(),
		"-an",
		"-r", f"{fps:.9f}",
		"-pix_fmt", "yuv420p",
		"-c:v", "libx264",
		"-crf", "18",
		str(output_path),
	]
	run_checked(cmd)


def decode_first_frame(video_path: Path) -> Image.Image:
	"""Decode the first RGB frame from a video file."""

	for frame in iio.imiter(video_path):
		return Image.fromarray(frame).convert("RGB")
	raise RejectedSample(f"No frames decoded from {video_path}")


def center_crop_square_image(image: Image.Image) -> Image.Image:
	"""Return the largest centered square crop of an image."""

	width, height = image.size
	side = min(width, height)
	left = (width - side) // 2
	top = (height - side) // 2
	return image.crop((left, top, left + side, top + side))


def image_mask_valid_fraction(mask_path: Path | None) -> float:
	"""Compute one DSLR reference mask valid-pixel fraction."""

	if mask_path is None or not mask_path.exists():
		return 1.0
	with Image.open(mask_path) as image:
		mask = np.asarray(center_crop_square_image(image.convert("L")))
	if mask.ndim == 3:
		mask = np.max(mask[..., :3], axis=-1)
	return float((mask > 127).mean())


# TODO: As mentioned above, use python interfaces rather than CLI.
def stream_mask_fraction(
	mask_path: Path,
	start_frame: int,
	fps: float,
	duration_s: float,
	crop_side: int,
	max_frames: int | None = None,
) -> float:
	"""Stream a cropped mask clip through FFmpeg and compute valid pixels."""

	if not mask_path.exists():
		return 1.0
	start_time = start_frame / fps
	cmd = [
		"ffmpeg", "-hide_banner", "-loglevel", "error",
		"-ss", f"{start_time:.9f}",
		"-i", str(mask_path),
		"-t", f"{duration_s:.9f}",
		"-vf", f"{ffmpeg_crop_expr()},format=gray",
		"-f", "rawvideo",
		"-pix_fmt", "gray",
		"pipe:1",
	]
	process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	assert process.stdout is not None
	frame_size = crop_side * crop_side
	valid_pixels = 0
	total_pixels = 0
	frames_seen = 0
	while True:
		if max_frames is not None and frames_seen >= max_frames:
			break
		chunk = process.stdout.read(frame_size)
		if not chunk:
			break
		if len(chunk) != frame_size:
			break
		mask = np.frombuffer(chunk, dtype=np.uint8)
		valid_pixels += int((mask > 127).sum())
		total_pixels += mask.size
		frames_seen += 1
	_, stderr = process.communicate()
	if process.returncode not in {0, None} and total_pixels == 0:
		raise RuntimeError(stderr.decode("utf-8", errors="replace"))
	if total_pixels == 0:
		return 1.0
	return valid_pixels / total_pixels


def load_dslr_candidates(scene: ScannetppScene_Release) -> list[DslrCandidate]:
	"""Load DSLR reference candidates that are not marked as bad."""

	transform_path = scene.dslr_nerfstudio_transform_undistorted_path
	if not transform_path.exists():
		raise RejectedSample(f"Missing DSLR transforms: {transform_path}")
	payload = read_json(transform_path)
	frames = list(payload.get("frames") or []) + list(payload.get("test_frames") or [])
	candidates: dict[str, DslrCandidate] = {}
	for frame in frames:
		if frame.get("is_bad", False):
			continue
		file_value = frame.get("file_path") or frame.get("image_path") or frame.get("name")
		if not file_value:
			continue
		image_path = Path(file_value)
		if not image_path.is_absolute():
			if image_path.exists():
				image_path = image_path
			else:
				image_path = scene.dslr_dir / image_path
				if not image_path.exists():
					image_path = scene.dslr_resized_undistorted_dir / Path(file_value).name
		if not image_path.exists():
			continue

		mask_value = frame.get("mask_path") or frame.get("mask")
		mask_path = None
		if mask_value:
			mask_candidate = Path(mask_value)
			if not mask_candidate.is_absolute():
				mask_candidate = scene.dslr_dir / mask_candidate
			mask_path = mask_candidate if mask_candidate.exists() else None
		if mask_path is None:
			default_mask = scene.dslr_resized_undistorted_mask_dir / f"{image_path.stem}.png"
			mask_path = default_mask if default_mask.exists() else None

		ref_id = f"dslr:{image_path.name}"
		candidates[ref_id] = DslrCandidate(image_path=image_path, mask_path=mask_path, ref_id=ref_id)
	return [candidates[key] for key in sorted(candidates)]


def deterministic_ref_hash(ref_sources: list[RefSource]) -> str:
	"""Hash the final fixed reference order into an eight-character suffix."""

	payload = "|".join(source.ref_id for source in ref_sources)
	return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def relative_sample_path(sample_id: str, *parts: str) -> str:
	"""Build a POSIX relative path inside the output sample tree."""

	return str(Path("samples") / sample_id / Path(*parts)).replace(os.sep, "/")


def ordinal(index: int) -> str:
	"""Return a compact English ordinal for reference-image prompts."""

	names = {
		1: "the first image",
		2: "the second image",
		3: "the third image",
		4: "the fourth image",
		5: "the fifth image",
		6: "the sixth image",
		7: "the seventh image",
		8: "the eighth image",
		9: "the ninth image",
		10: "the tenth image",
	}
	return names.get(index, f"image {index}")


class DataSamplingStage:
	"""Sample ScanNet++ clips and reference images into an atomic temp folder."""

	def __init__(
		self,
		args: argparse.Namespace,
		scene_ids: list[str],
		scene_types: dict[str, str],
		rng: random.Random,
	):
		self.args = args
		self.scene_ids = scene_ids
		self.scene_types = scene_types
		self.rng = rng
		self.data_root = Path(args.scannetpp_root) / "data"
		self.samples_root = output_root(args) / "samples"

	# TODO: I hate having to handle the start frame separately everytime, for example, calling `stream_mask_fraction` just to get the mask of the start frame. It results in duplicated code and reduces readability. Think of a clever way to maximize shared logic, try to let all reference images share the logic. Afterall, the start frame has no difference from other DSLR images, except that it is specially marked. You may even store start frame temporarily on disk once, so that it can be treated similarly with other DSLR images, but if you have cleverer alternative, I'd like to have them.
	
	def _select_references(
		self,
		scene: ScannetppScene_Release,
		start_frame: int,
	) -> list[RefSource]:
		"""Select and shuffle start-frame and DSLR reference sources."""

		num_refs = self.rng.randint(1, self.args.max_ref_images)
		include_start = self.rng.random() < self.args.include_start_frame_prob
		dslr_needed = num_refs - int(include_start)
		candidates = load_dslr_candidates(scene)
		if len(candidates) < dslr_needed:
			raise RejectedSample(
				f"Scene {scene.scene_id} has {len(candidates)} usable DSLR images, needs {dslr_needed}."
			)
		ref_sources: list[RefSource] = []
		if include_start:
			ref_sources.append(RefSource(
				kind="iphone_start",
				ref_id=f"iphone_start:f{start_frame:07d}",
				image_path=None,
				mask_path=None,
				is_start_frame=True,
			))
		for candidate in self.rng.sample(candidates, k=dslr_needed):
			ref_sources.append(RefSource(
				kind="dslr",
				ref_id=candidate.ref_id,
				image_path=candidate.image_path,
				mask_path=candidate.mask_path,
				is_start_frame=False,
			))
		self.rng.shuffle(ref_sources)
		return ref_sources

	def _write_references(
		self,
		ctx: SampleContext,
		ref_sources: list[RefSource],
		start_image: Image.Image,
		start_valid_fraction: float,
	) -> list[dict[str, Any]]:
		"""Copy selected references into fixed sample order."""

		ref_dir = ctx.tmp_dir / "ref_imgs"
		ref_dir.mkdir(parents=True, exist_ok=True)
		metadata: list[dict[str, Any]] = []
		for index, source in enumerate(ref_sources, start=1):
			if source.is_start_frame:
				image = start_image.copy()
				valid_fraction = start_valid_fraction
			else:
				if source.image_path is None:
					raise RejectedSample("Non-start reference has no image path.")
				with Image.open(source.image_path) as loaded:
					image = center_crop_square_image(loaded.convert("RGB"))
				valid_fraction = image_mask_valid_fraction(source.mask_path)
			if self.args.filter_pixel_valid_fraction_min is not None and valid_fraction < self.args.filter_pixel_valid_fraction_min:
				raise RejectedSample(
					f"Reference valid fraction {valid_fraction:.4f} below threshold."
				)
			output_name = f"ref_{index:02d}.jpg"
			output_path = ref_dir / output_name
			image.save(output_path, quality=95)
			width, height = image.size
			metadata.append({
				"index": index,
				"is_start_frame": bool(source.is_start_frame),
				"width": width,
				"height": height,
				"path": relative_sample_path(ctx.sample_id, "ref_imgs", output_name),
			})
		return metadata

	def run(self) -> SampleContext:
		"""Create one sampled candidate under `samples/.tmp_<sample_id>`."""

		scene_id = self.rng.choice(self.scene_ids)
		scene = ScannetppScene_Release(scene_id, data_root=self.data_root)
		if not scene.iphone_video_path.exists():
			raise RejectedSample(f"Missing iPhone RGB video: {scene.iphone_video_path}")
		video_info = probe_video(scene.iphone_video_path)
		clip_frames = int(round(video_info.fps * self.args.clip_seconds))
		if clip_frames <= 1 or video_info.num_frames <= clip_frames:
			raise RejectedSample(f"Video too short for {self.args.clip_seconds}s clip: {scene.iphone_video_path}")
		start_frame = self.rng.randint(0, video_info.num_frames - clip_frames)
		ref_sources = self._select_references(scene, start_frame)
		ref_hash = deterministic_ref_hash(ref_sources)
		sample_id = f"{scene_id}__f{start_frame:07d}__r{ref_hash}"
		tmp_dir = self.samples_root / f".tmp_{sample_id}"
		final_dir = self.samples_root / sample_id
		if final_dir.exists() or tmp_dir.exists():
			raise RejectedSample(f"Sample already exists or is in progress: {sample_id}")

		tmp_dir.mkdir(parents=True)
		(tmp_dir / "intermediate").mkdir()
		try:
			gt_clip_path = tmp_dir / "gt_clip.mp4"
			extract_square_clip(
				scene.iphone_video_path,
				gt_clip_path,
				start_frame=start_frame,
				fps=video_info.fps,
				duration_s=self.args.clip_seconds,
			)
			start_image = decode_first_frame(gt_clip_path)
			clip_width, clip_height = start_image.size
			video_valid_fraction = stream_mask_fraction(
				scene.iphone_video_mask_path,
				start_frame=start_frame,
				fps=video_info.fps,
				duration_s=self.args.clip_seconds,
				crop_side=min(video_info.width, video_info.height),
			)
			if self.args.filter_pixel_valid_fraction_min is not None and video_valid_fraction < self.args.filter_pixel_valid_fraction_min:
				raise RejectedSample(f"Video valid fraction {video_valid_fraction:.4f} below threshold.")
			start_valid_fraction = stream_mask_fraction(
				scene.iphone_video_mask_path,
				start_frame=start_frame,
				fps=video_info.fps,
				duration_s=1.0 / video_info.fps,
				crop_side=min(video_info.width, video_info.height),
				max_frames=1,
			)
			ctx = SampleContext(
				scene_id=scene_id,
				scene_type=str(self.scene_types.get(scene_id, "unknown")),
				scene=scene,
				sample_id=sample_id,
				start_frame=start_frame,
				clip_frames=clip_frames,
				video_fps=video_info.fps,
				tmp_dir=tmp_dir,
				final_dir=final_dir,
				manifest_entry={},
			)
			ref_metadata = self._write_references(ctx, ref_sources, start_image, start_valid_fraction)
			ctx.manifest_entry = {
				"sample_id": sample_id,
				"scene_id": scene_id,
				"scene_type": ctx.scene_type,
				"split": self.args.split,
				"ref_imgs": ref_metadata,
				"gt_clip": {
					"fps": video_info.fps,
					"duration_sec": self.args.clip_seconds,
					"width": clip_width,
					"height": clip_height,
					"path": relative_sample_path(sample_id, "gt_clip.mp4"),
				},
				"synthesized_prompt": "",
				"distilled_prompts": {
					"medium": "",
					"coarse": "",
				},
			}
			write_json(ctx.intermediate_dir / "sampling_metadata.json", {
				"source_scene_id": scene_id,
				"start_frame": start_frame,
				"clip_frames": clip_frames,
				"video_valid_fraction": video_valid_fraction,
				"start_frame_valid_fraction": start_valid_fraction,
				"reference_sources": [
					{
						"index": index,
						"ref_id": source.ref_id,
						"kind": source.kind,
						"is_start_frame": source.is_start_frame,
					}
					for index, source in enumerate(ref_sources, start=1)
				],
			})
			return ctx
		except Exception:
			shutil.rmtree(tmp_dir, ignore_errors=True)
			raise


def parse_pose_matrix(value: Any) -> np.ndarray | None:
	"""Convert one pose payload into a finite 4x4 matrix."""

	if value is None:
		return None
	if isinstance(value, dict):
		for key in ("transform_matrix", "pose", "matrix", "c2w"):
			if key in value:
				value = value[key]
				break
	array = np.asarray(value, dtype=np.float64)
	if array.shape != (4, 4) or not np.isfinite(array).all():
		return None
	return array


def load_pose_sequence(path: Path) -> list[np.ndarray | None]:
	"""Load aligned iPhone poses, falling back to raw poses."""

	payload = read_json(path)
	poses_payload = payload.get("aligned_poses")
	if poses_payload is None:
		poses_payload = payload.get("poses")
	if poses_payload is None:
		raise RejectedSample(f"No poses found in {path}")
	if isinstance(poses_payload, dict):
		def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
			key = item[0]
			return (int(key), key) if str(key).isdigit() else (10**12, str(key))
		values = [value for _, value in sorted(poses_payload.items(), key=sort_key)]
	else:
		values = list(poses_payload)
	return [parse_pose_matrix(value) for value in values]


def rotation_to_yaw_pitch_roll(rotation: np.ndarray) -> tuple[float, float, float]:
	"""Return approximate yaw, pitch, and roll in degrees for camera axes."""

	forward = rotation @ np.array([0.0, 0.0, 1.0])
	down = rotation @ np.array([0.0, 1.0, 0.0])
	yaw = math.degrees(math.atan2(forward[0], forward[2]))
	pitch = math.degrees(math.atan2(-forward[1], math.hypot(forward[0], forward[2])))
	roll = math.degrees(math.atan2(-down[0], down[1]))
	return yaw, pitch, roll


def round_float(value: float, digits: int = 4) -> float:
	"""Round floats for stable, compact JSON metadata."""

	return round(float(value), digits)


# TODO: Rename this stage from `TrajectoryStatisticsStage` to `MotionExtractionStage`. This is more than renaming the class, it involves the change of terms for the entire codebase, including docs, file names, prompt templates, and more. MotionExtraction here means extracting motion units from poses.
class TrajectoryStatisticsStage:
	"""Convert iPhone poses into whole-clip and sub-trajectory statistics."""

	def __init__(self, args: argparse.Namespace):
		self.args = args

	def _valid_poses(self, poses: list[np.ndarray]) -> list[np.ndarray]:
		"""Return a valid pose sequence."""
		
		valid = [pose for pose in poses if pose is not None]
		if len(valid) < 2:
			raise RejectedSample("Fewer than two valid poses in sampled clip.")
		valid_fraction = len(valid) / len(poses)
		if self.args.filter_pose_valid_fraction_min is not None and valid_fraction < self.args.filter_pose_valid_fraction_min:
			raise RejectedSample(f"Pose valid fraction {valid_fraction:.4f} below threshold.")
		return valid
	
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
		yaw, pitch, roll = rotation_to_yaw_pitch_roll(relative[:3, :3])
		
		return {
			"duration_s": round_float(duration_s),
			f"{prefix}trajectory_length_m": round_float(trajectory_length),
			f"{prefix}translation_right_m": round_float(translation[0]),
			f"{prefix}translation_down_m": round_float(translation[1]),
			f"{prefix}translation_forward_m": round_float(translation[2]),
			f"{prefix}translation_distance_m": round_float(float(np.linalg.norm(translation))),
			f"{prefix}delta_yaw_deg": round_float(yaw),
			f"{prefix}delta_pitch_deg": round_float(pitch),
			f"{prefix}delta_roll_deg": round_float(roll),
		}

	def run(self, ctx: SampleContext) -> None:
		"""Compute and save `trajectory_statistics.json`."""

		all_poses = load_pose_sequence(ctx.scene.iphone_pose_intrinsic_imu_path)
		clip_poses = all_poses[ctx.start_frame:(ctx.start_frame + ctx.clip_frames)]
		if len(clip_poses) < ctx.clip_frames:
			raise RejectedSample("Pose sequence is shorter than the sampled clip.")
		
		clip_valid_poses = self._valid_poses(clip_poses)
		trajectory = self._motion_stats(
			poses=clip_valid_poses,
			duration_s=self.args.clip_seconds,
		)
		trajectory_length = trajectory["trajectory_length_m"]
		if self.args.filter_camera_trajectory_length_m_min is not None and trajectory_length < self.args.filter_camera_trajectory_length_m_min:
			raise RejectedSample(f"Trajectory length {trajectory_length:.4f} below minimum.")
		if self.args.filter_camera_trajectory_length_m_max is not None and trajectory_length > self.args.filter_camera_trajectory_length_m_max:
			raise RejectedSample(f"Trajectory length {trajectory_length:.4f} above maximum.")

		first_pose = clip_valid_poses[0]
		sub_trajectories: list[dict[str, Any]] = []
		unit = self.args.llm_trajectory_digest_unit_seconds
		num_units = int(math.ceil(self.args.clip_seconds / unit))
		for unit_index in range(num_units):
			start_time = unit_index * unit
			end_time = min((unit_index + 1) * unit, self.args.clip_seconds)
			start_index = int(round(start_time * ctx.video_fps))
			end_index = min(int(round(end_time * ctx.video_fps)), len(clip_poses))
			subclip_poses = clip_poses[start_index:end_index]
			
			subclip_valid_poses = self._valid_poses(subclip_poses)
			subtrajectory = self._motion_stats(
				poses=subclip_valid_poses,
				duration_s=end_time - start_time,
				prefix="local_"
			)
			if subclip_valid_poses:
				relative_first = np.linalg.inv(first_pose) @ subclip_valid_poses[0]
				first_yaw, first_pitch, first_roll = rotation_to_yaw_pitch_roll(relative_first[:3, :3])
				first_position = [round_float(value) for value in relative_first[:3, 3].tolist()]
			else:
				first_yaw, first_pitch, first_roll = 0.0, 0.0, 0.0
				first_position = [0.0, 0.0, 0.0]
			sub_trajectories.append({
				"index": unit_index,
				"time_range_s": [round_float(start_time), round_float(end_time)],
				"first_pose_position_m": first_position,
				"first_pose_yaw_deg": round_float(first_yaw),
				"first_pose_pitch_deg": round_float(first_pitch),
				"first_pose_roll_deg": round_float(first_roll),
				**subtrajectory,
			})
		write_json(ctx.intermediate_dir / "trajectory_statistics.json", {
			"trajectory": trajectory,
			"sub_trajectories": sub_trajectories,
		})


def extract_json_response(text: str) -> Any:
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


# TODO: DThere is no need for a separate function for it, delete it.
def local_media_uri(path: Path) -> str:
	"""Return a file URI for local media paths."""

	return path.resolve().as_uri()


class TextGenerationBackend:
	"""Hugging Face causal LLM backend with JSON retry/repair support."""

	def __init__(self, model_path: str, local_rank: int | None):
		# TODO: why don't you import at the top of this file?
		import torch
		from transformers import AutoModelForCausalLM, AutoTokenizer

		# TODO: This is absurd, just use torch, why store that as a property?
		self.torch = torch
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
		if torch.cuda.is_available() and local_rank is not None:
			load_kwargs["device_map"] = {"": local_rank}
		self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
		if "device_map" not in load_kwargs:
			self.model.to("cuda" if torch.cuda.is_available() else "cpu")
		self.model.eval()

	def _device(self):
		"""Return the first model parameter device."""

		return next(self.model.parameters()).device

	def generate(self, prompt: str, temperature: float, max_new_tokens: int, media: list[dict[str, Any]] | None = None) -> str:
		"""Generate text from a prompt."""

		del media
		messages = [{"role": "user", "content": prompt}]
		text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		inputs = self.tokenizer([text], return_tensors="pt").to(self._device())
		generation_kwargs = {
			"max_new_tokens": max_new_tokens,
			"do_sample": temperature > 0,
			"pad_token_id": self.tokenizer.eos_token_id,
		}
		if temperature > 0:
			generation_kwargs["temperature"] = temperature
		with self.torch.no_grad():
			outputs = self.model.generate(**inputs, **generation_kwargs)
		new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
		return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

	def generate_json(
		self,
		prompt: str,
		temperature: float,
		max_new_tokens: int,
		retries: int,
		media: list[dict[str, Any]] | None = None,
	) -> Any:
		"""Generate and parse JSON, retrying with repair prompts when necessary."""

		# TODO: There is no need for repetitive repairs, retry only once. If the first JSON decoding of the normal output fails, then repair the output with deterministic decoding (temperature 0.0). If the second JSON decoding fails again, then simply reject the sample.
		output = ""
		current_prompt = prompt
		for attempt in range(retries + 1):
			output = self.generate(current_prompt, temperature=temperature, max_new_tokens=max_new_tokens, media=media)
			try:
				return extract_json_response(output)
			except Exception:
				if attempt >= retries:
					raise
				current_prompt = (
					"Repair the following model response into valid JSON only. "
					"Do not add Markdown or explanation.\n\n"
					f"Original task:\n{prompt}\n\nInvalid response:\n{output}"
				)
		raise RuntimeError(f"Failed to parse JSON response: {output}")


class VisionLanguageBackend(TextGenerationBackend):
	"""Hugging Face image/video-to-text backend for Qwen3-VL-style models."""

	def __init__(self, model_path: str, local_rank: int | None):
		# TODO: Just import at the top of the file, ok? Use torch throughout this file boldly, it is absurd to write `self.torch`.
		import torch
		from transformers import AutoModelForImageTextToText, AutoProcessor

		self.torch = torch
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
		if torch.cuda.is_available() and local_rank is not None:
			load_kwargs["device_map"] = {"": local_rank}
		self.model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
		if "device_map" not in load_kwargs:
			self.model.to("cuda" if torch.cuda.is_available() else "cpu")
		self.model.eval()

	def generate(self, prompt: str, temperature: float, max_new_tokens: int, media: list[dict[str, Any]] | None = None) -> str:
		"""Generate text from optional image/video media plus a prompt."""

		content: list[dict[str, Any]] = []
		for item in media or []:
			if item["type"] == "image":
				content.append({
					"type": "image",
					"image": local_media_uri(Path(item["path"])),
					"resized_width": int(item["resized_width"]),
					"resized_height": int(item["resized_height"]),
				})
			elif item["type"] == "video":
				content.append({
					"type": "video",
					"video": local_media_uri(Path(item["path"])),
					"fps": float(item["fps"]),
					"resized_width": int(item["resized_width"]),
					"resized_height": int(item["resized_height"]),
				})
		content.append({"type": "text", "text": prompt})
		messages = [{"role": "user", "content": content}]
		text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

		inputs = self.processor(
			text=[text],
			images=image_inputs,
			videos=video_inputs,
			padding=True,
			return_tensors="pt",
			**video_kwargs,
		).to(self._device())
		generation_kwargs = {
			"max_new_tokens": max_new_tokens,
			"do_sample": temperature > 0,
		}
		if temperature > 0:
			generation_kwargs["temperature"] = temperature
		with self.torch.no_grad():
			outputs = self.model.generate(**inputs, **generation_kwargs)
		new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
		return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# TODO: Rename this stage from `TrajectoryDigestStage` to `MotionDigestingStage`. This is more than renaming the class, it involves the change of terms for the entire codebase, including docs,file names, prompt templates, and more. MotionDigesting here means digesting the numeric motion units.
class TrajectoryDigestStage:
	"""Use an LLM to convert numeric trajectory statistics into a caption."""

	def __init__(self, args: argparse.Namespace, llm: TextGenerationBackend):
		self.args = args
		self.llm = llm

	def run(self, ctx: SampleContext) -> None:
		"""Save `trajectory_caption.json`."""

		stats = read_json(ctx.intermediate_dir / "trajectory_statistics.json")
		prompt = TRAJECTORY_DIGEST_TEMPLATE.replace("<TRAJECTORY_STATISTICS_JSON>", json.dumps(stats, ensure_ascii=False, indent=2))
		result = self.llm.generate_json(
			prompt,
			temperature=self.args.llm_trajectory_digest_temperature,
			max_new_tokens=self.args.llm_rewriting_max_new_tokens,
			retries=self.args.json_retries,
		)
		write_json(ctx.intermediate_dir / "trajectory_caption.json", {
			"trajectory_caption": str(result["trajectory_caption"]),
		})


# TODO: Rename this stage from `VideoCaption` to `VideoCaptioning`. This is more than renaming the class, it involves the change of terms for the entire codebase, including docs, file names, prompt templates, and more.
class VideoCaptionStage:
	"""Use a VLM to caption the ground-truth video with trajectory guidance."""

	def __init__(self, args: argparse.Namespace, vlm: VisionLanguageBackend):
		self.args = args
		self.vlm = vlm

	def run(self, ctx: SampleContext) -> None:
		"""Save `video_caption.json`."""

		trajectory_caption = read_json(ctx.intermediate_dir / "trajectory_caption.json")
		prompt = VIDEO_CAPTION_TEMPLATE.replace("<TRAJECTORY_CAPTION_JSON>", json.dumps(trajectory_caption, ensure_ascii=False, indent=2))
		result = self.vlm.generate_json(
			prompt,
			temperature=self.args.vlm_captioning_temperature,
			max_new_tokens=self.args.vlm_captioning_max_new_tokens,
			retries=self.args.json_retries,
			media=[{
				"type": "video",
				"path": str(ctx.tmp_dir / "gt_clip.mp4"),
				"fps": self.args.vlm_captioning_video_fps,
				"resized_width": self.args.vlm_captioning_video_width,
				"resized_height": self.args.vlm_captioning_video_height,
			}],
		)
		write_json(ctx.intermediate_dir / "video_caption.json", {
			"video_caption": str(result["video_caption"]),
		})


# TODO: Rename this stage from `ImageCaption` to `ImageCaptioning`. This is more than renaming the class, it involves the change of terms for the entire codebase, including docs, file names, prompt templates, and more.
class ImageCaptionStage:
	"""Use a VLM to caption all ordered reference images."""

	def __init__(self, args: argparse.Namespace, vlm: VisionLanguageBackend):
		self.args = args
		self.vlm = vlm

	def run(self, ctx: SampleContext) -> None:
		"""Save `image_captions.json`."""

		captions: list[str] = []
		for ref in ctx.manifest_entry["ref_imgs"]:
			index = int(ref["index"])
			prompt = (
				IMAGE_CAPTION_TEMPLATE
				.replace("<REF_INDEX>", f"{ordinal(index)} (index {index})")
				.replace("<IS_VIDEO_START_CLAUSE>", "is" if ref["is_start_frame"] else "is not")
			)
			result = self.vlm.generate_json(
				prompt,
				temperature=self.args.vlm_captioning_temperature,
				max_new_tokens=self.args.vlm_captioning_max_new_tokens,
				retries=self.args.json_retries,
				media=[{
					"type": "image",
					"path": str(ctx.tmp_dir / "ref_imgs" / f"ref_{index:02d}.jpg"),
					"resized_width": self.args.vlm_captioning_image_width,
					"resized_height": self.args.vlm_captioning_image_height,
				}],
			)
			captions.append(str(result["image_caption"]))
		write_json(ctx.intermediate_dir / "image_captions.json", {
			"image_captions": captions,
		})


# TODO: Rename this stage from `CrossReferenceStage` to `CaptionWiringStage`. This is more than renaming the class, it involves the change of terms for the entire codebase, including docs, file names, prompt templates, and more. CaptionWiring here means properly hook up the video caption's connections to image captions.
class CrossReferenceStage:
	"""Use an LLM to cross-reference video and reference-image captions."""

	def __init__(self, args: argparse.Namespace, llm: TextGenerationBackend):
		self.args = args
		self.llm = llm

	def run(self, ctx: SampleContext) -> None:
		"""Save `cross_referenced_caption.json`."""

		video_caption = read_json(ctx.intermediate_dir / "video_caption.json")
		image_captions = read_json(ctx.intermediate_dir / "image_captions.json")
		prompt = (
			CROSS_REFERENCE_TEMPLATE
			.replace("<VIDEO_CAPTION_JSON>", json.dumps(video_caption, ensure_ascii=False, indent=2))
			.replace("<IMAGE_CAPTIONS_JSON>", json.dumps(image_captions, ensure_ascii=False, indent=2))
		)
		result = self.llm.generate_json(
			prompt,
			temperature=self.args.llm_rewriting_temperature,
			max_new_tokens=self.args.llm_rewriting_max_new_tokens,
			retries=self.args.json_retries,
		)
		write_json(ctx.intermediate_dir / "cross_referenced_caption.json", {
			"cross_referenced_caption": str(result["cross_referenced_caption"]),
		})


# TODO: Rename this stage from `PromptRewriteStage` to `CaptionRephrasingStage`. This is more than renaming the class, it involves the change of terms for the entire codebase, including docs, file names, prompt templates, and more. CaptionRephrasing here means rewriting captions to prompts.
class PromptRewriteStage:
	"""Use an LLM to rewrite the cross-referenced caption as a user prompt."""

	def __init__(self, args: argparse.Namespace, llm: TextGenerationBackend):
		self.args = args
		self.llm = llm

	def run(self, ctx: SampleContext) -> None:
		"""Update the manifest entry with `synthesized_prompt`."""

		cross_referenced = read_json(ctx.intermediate_dir / "cross_referenced_caption.json")
		prompt = PROMPT_REWRITE_TEMPLATE.replace("<CROSS_REFERENCED_CAPTION_JSON>", json.dumps(cross_referenced, ensure_ascii=False, indent=2))
		result = self.llm.generate_json(
			prompt,
			temperature=self.args.llm_rewriting_temperature,
			max_new_tokens=self.args.llm_rewriting_max_new_tokens,
			retries=self.args.json_retries,
		)
		ctx.manifest_entry["synthesized_prompt"] = str(result["synthesized_prompt"])


# TODO: Rename this stage from `JudgeStage` to `CriticJudgingStage`. This is more than renaming the class, it involves the change of terms for the entire codebase, including docs, prompt templates, and more. CriticJudging here means letting a LLM critic to judge the sample.
class JudgeStage:
	"""Use an LLM as the final prompt-validation gate."""

	def __init__(self, args: argparse.Namespace, llm: TextGenerationBackend):
		self.args = args
		self.llm = llm

	def run(self, ctx: SampleContext) -> None:
		"""Save `llm_judge.json` and reject invalid prompt candidates."""

		trajectory_caption = read_json(ctx.intermediate_dir / "trajectory_caption.json")
		video_caption = read_json(ctx.intermediate_dir / "video_caption.json")
		image_captions = read_json(ctx.intermediate_dir / "image_captions.json")
		prompt = (
			JUDGE_TEMPLATE
			.replace("<TRAJECTORY_CAPTION_JSON>", json.dumps(trajectory_caption, ensure_ascii=False, indent=2))
			.replace("<VIDEO_CAPTION_JSON>", json.dumps(video_caption, ensure_ascii=False, indent=2))
			.replace("<IMAGE_CAPTIONS_JSON>", json.dumps(image_captions, ensure_ascii=False, indent=2))
			.replace("<SYNTHESIZED_PROMPT>", ctx.manifest_entry["synthesized_prompt"])
		)
		result = self.llm.generate_json(
			prompt,
			temperature=self.args.llm_rewriting_temperature,
			max_new_tokens=self.args.llm_rewriting_max_new_tokens,
			retries=self.args.json_retries,
		)
		fatal_checks = result.get("fatal_checks") or {}
		quality_checks = result.get("quality_checks") or {}
		fatal_passed = all(bool(value) for value in fatal_checks.values()) and len(fatal_checks) > 0
		quality_values = [bool(value) for value in quality_checks.values()]
		quality_score = sum(quality_values) / len(quality_values) if quality_values else 0.0
		result["computed_quality_score"] = quality_score
		write_json(ctx.intermediate_dir / "llm_judge.json", result)
		if not fatal_passed:
			raise RejectedSample("LLM judge fatal checks failed.")
		if self.args.filter_llm_judge_quality_score is not None and quality_score < self.args.filter_llm_judge_quality_score:
			raise RejectedSample(f"LLM judge quality score {quality_score:.4f} below threshold.")


class DistillationStage:
	"""Use an LLM to create medium and coarse prompt variants."""

	def __init__(self, args: argparse.Namespace, llm: TextGenerationBackend):
		self.args = args
		self.llm = llm

	def run(self, ctx: SampleContext) -> None:
		"""Update the manifest entry with distilled prompt variants."""

		prompt = DISTILLATION_TEMPLATE.replace("<SYNTHESIZED_PROMPT>", ctx.manifest_entry["synthesized_prompt"])
		result = self.llm.generate_json(
			prompt,
			temperature=self.args.llm_rewriting_temperature,
			max_new_tokens=self.args.llm_rewriting_max_new_tokens,
			retries=self.args.json_retries,
		)
		ctx.manifest_entry["distilled_prompts"] = {
			"medium": str(result["medium_prompt"]),
			"coarse": str(result["coarse_prompt"]),
		}


def commit_sample(args: argparse.Namespace, ctx: SampleContext) -> bool | None:
	"""Atomically publish a sample and update the manifest ledger.

	Returns:
		`True` when committed, `False` for a duplicate, and `None` when the
		global target has already been reached.
	"""

	with exclusive_lock(lock_path(args)):
		state = read_state(args)
		if state["current_count"] >= state["target_count"]:
			shutil.rmtree(ctx.tmp_dir, ignore_errors=True)
			return None
		if ctx.final_dir.exists():
			shutil.rmtree(ctx.tmp_dir, ignore_errors=True)
			return False
		write_json(ctx.tmp_dir / "sample.json", ctx.manifest_entry)
		os.replace(ctx.tmp_dir, ctx.final_dir)
		append_jsonl(manifest_path(args), ctx.manifest_entry)
		state["current_count"] += 1
		write_json_atomic(state_path(args), state)
		return True


def process_seed(args: argparse.Namespace, worker_index: int) -> int:
	"""Create a process-local time-dependent seed."""

	base = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), "little")
	return (base ^ time.time_ns() ^ (os.getpid() << 16) ^ worker_index) & 0xFFFFFFFF
	

def run_worker(args: argparse.Namespace, worker_index: int = 0) -> None:
	"""Run one independent sample-generation loop."""

	if not torch.cuda.is_available():
		local_rank = None
	device_count = torch.cuda.device_count()
	if device_count <= 0:
		local_rank = None
	local_rank = worker_index % device_count
	torch.cuda.set_device(local_rank)
	
	seed = process_seed(args, worker_index)
	rng = random.Random(seed)
	np.random.seed(seed)
	scene_ids = load_scene_ids(args)
	scene_types = load_scene_types(Path(args.scannetpp_root))
	sampler = DataSamplingStage(args, scene_ids, scene_types, rng)
	trajectory_stage = TrajectoryStatisticsStage(args)
	state = read_state(args)
	if state["current_count"] >= state["target_count"]:
		return

	# TODO: Currently, it seems that wach process load the two gaint backbones to the target GPU, while only one of the model is used at each stage. Can you implement some configurable GPU VRAM optimization techniques, such as optional CPU offloading, to mitigate CUDA OOM if I ever encounter.
	print(f"[worker {worker_index}] loading VLM from {args.vlm_backend_path}", flush=True)
	vlm = VisionLanguageBackend(args.vlm_backend_path, local_rank=local_rank)
	print(f"[worker {worker_index}] loading LLM from {args.llm_backend_path}", flush=True)
	llm = TextGenerationBackend(args.llm_backend_path, local_rank=local_rank)
	stages: list[Callable[[SampleContext], None]] = [
		trajectory_stage.run,
		TrajectoryDigestStage(args, llm).run,
		VideoCaptionStage(args, vlm).run,
		ImageCaptionStage(args, vlm).run,
		CrossReferenceStage(args, llm).run,
		PromptRewriteStage(args, llm).run,
		JudgeStage(args, llm).run,
		DistillationStage(args, llm).run,
	]

	while True:
		state = read_state(args)
		if state["current_count"] >= state["target_count"]:
			break
		ctx: SampleContext | None = None
		try:
			ctx = sampler.run()
			for stage in stages:
				stage(ctx)
			result = commit_sample(args, ctx)
			if result is None:
				break
			if result:
				updated_state = read_state(args)
				print(
					f"[worker {worker_index}] committed {ctx.sample_id} "
					f"({updated_state['current_count']}/{updated_state['target_count']})",
					flush=True,
				)
		except RejectedSample as error:
			if ctx is not None:
				shutil.rmtree(ctx.tmp_dir, ignore_errors=True)
			print(f"[worker {worker_index}] rejected sample: {error}", flush=True)
		except Exception:
			if ctx is not None:
				shutil.rmtree(ctx.tmp_dir, ignore_errors=True)
			raise


def main() -> None:
	"""Launch one or more independent curation workers."""

	args = parse_args()
	initialize_state(args)
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
