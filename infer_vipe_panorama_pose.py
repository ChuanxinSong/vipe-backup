from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infer_vipe_panorama_pose")

REPO_ROOT = Path(__file__).resolve().parent
POSE_RENDER_I2V_MOGE_SEGMENT_FIRST_FORMAT_VERSION = "pose_render_i2v_moge_segment_first_v1"
POSE_I2V_COMPARISON_SEPARATOR_HEIGHT = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate and save ViPE poses for panorama image sequences.")
    parser.add_argument("--json_path", type=Path, required=True, help="Input JSON describing panorama clips.")
    parser.add_argument(
        "--dataset_format",
        choices=["pvdepth_json", "omniroam_png", "omniroam_h5", "pose_i2v_results"],
        default="pvdepth_json",
        help="Input dataset layout.",
    )
    parser.add_argument(
        "--image_base_dir",
        type=Path,
        default=None,
        help="Root directory for pvdepth rgb_path values or OmniRoam PNG scenes.",
    )
    parser.add_argument("--h5_data_root", type=Path, default=None, help="Root directory for OmniRoam H5 files.")
    parser.add_argument(
        "--pose_i2v_results_root",
        type=Path,
        default=None,
        help="Root containing <scene_id>/metadata.json and segment comparison PNGs.",
    )
    parser.add_argument(
        "--pose_i2v_scope",
        choices=["first_segment", "all_segments"],
        default="first_segment",
        help="Frames selected from pose-I2V results.",
    )
    parser.add_argument(
        "--pose_i2v_expected_segments",
        type=int,
        default=8,
        help="Required segment count for pose_i2v_scope=all_segments.",
    )
    parser.add_argument("--split_subset", default="test", help="Split key used by OmniRoam split JSON.")
    parser.add_argument("--interiorgs_frames_subdir", default="pano_camera0", help="OmniRoam PNG frame subdirectory.")
    parser.add_argument("--interiorgs_max_frames", type=int, default=800, help="Frames per OmniRoam scene.")
    parser.add_argument("--interiorgs_frame_ext", default="png", help="OmniRoam PNG frame extension.")
    parser.add_argument(
        "--output_root_dir",
        type=Path,
        default=Path("carla_benchmark_results/vipe_pano_pose"),
        help="Root directory for pose outputs.",
    )
    parser.add_argument("--resolution", type=int, default=1024, help="Panorama width; height is width / 2.")
    parser.add_argument(
        "--virtual_view_height",
        type=int,
        default=None,
        help=(
            "Virtual pinhole view height. Defaults to resolution/4 for legacy inputs and 256 for "
            "pose_i2v_results."
        ),
    )
    parser.add_argument("--start_clip_idx", type=int, default=0, help="First loaded clip index to process.")
    parser.add_argument("--num_clips", type=int, default=0, help="Maximum clips to process; 0 means all remaining clips.")
    parser.add_argument(
        "--cuda_reserve_gib",
        type=float,
        default=0.0,
        help=(
            "Target total CUDA memory, in GiB, to reserve in the PyTorch caching allocator before inference. "
            "Zero disables early reservation."
        ),
    )
    parser.add_argument(
        "--cuda_reserve_safety_gib",
        type=float,
        default=2.0,
        help="Minimum CUDA memory, in GiB, that must remain globally free after early reservation.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-estimate clips whose pose output already exists.")
    return parser.parse_args()


def reserve_cuda_memory(target_gib: float, safety_gib: float = 2.0) -> dict[str, float]:
    """Populate PyTorch's CUDA cache up to a target total reservation.

    The temporary tensor is deleted but the default caching allocator retains its
    CUDA allocation for reuse by later model and factor-graph tensors. Callers
    must not run ``torch.cuda.empty_cache()`` after this function if they want to
    keep the early reservation.
    """
    if not math.isfinite(target_gib) or target_gib < 0:
        raise ValueError("--cuda_reserve_gib must be finite and non-negative.")
    if not math.isfinite(safety_gib) or safety_gib < 0:
        raise ValueError("--cuda_reserve_safety_gib must be finite and non-negative.")
    if target_gib == 0:
        return {
            "target_gib": 0.0,
            "reserved_before_gib": 0.0,
            "reserved_after_gib": 0.0,
            "free_after_gib": 0.0,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA reservation was requested, but torch.cuda.is_available() is false.")

    gib = 1024**3
    target_bytes = int(target_gib * gib)
    safety_bytes = int(safety_gib * gib)

    torch.cuda.init()
    device = torch.cuda.current_device()
    reserved_before = torch.cuda.memory_reserved(device)
    allocated_before = torch.cuda.memory_allocated(device)
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    reserve_bytes = max(0, target_bytes - reserved_before)

    if reserve_bytes > max(0, free_before - safety_bytes):
        raise RuntimeError(
            "Cannot satisfy requested CUDA reservation: "
            f"target={target_gib:.2f} GiB, already_reserved={reserved_before / gib:.2f} GiB, "
            f"globally_free={free_before / gib:.2f} GiB, safety={safety_gib:.2f} GiB, "
            f"device={device}. Reduce --cuda_reserve_gib or use a less occupied GPU."
        )

    if reserve_bytes > 0:
        reservation = torch.empty(reserve_bytes, dtype=torch.uint8, device=torch.device("cuda", device))
        torch.cuda.synchronize(device)
        del reservation

    reserved_after = torch.cuda.memory_reserved(device)
    allocated_after = torch.cuda.memory_allocated(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    if reserved_after < target_bytes:
        raise RuntimeError(
            "CUDA allocation completed, but PyTorch did not retain the requested cache: "
            f"target={target_gib:.2f} GiB, retained={reserved_after / gib:.2f} GiB. "
            "Check that PYTORCH_NO_CUDA_MEMORY_CACHING is not set and that the active allocator retains "
            "inactive blocks."
        )
    result = {
        "target_gib": target_gib,
        "reserved_before_gib": reserved_before / gib,
        "reserved_after_gib": reserved_after / gib,
        "allocated_before_gib": allocated_before / gib,
        "allocated_after_gib": allocated_after / gib,
        "free_after_gib": free_after / gib,
        "total_gib": total_bytes / gib,
    }
    logger.info(
        "CUDA early reservation on logical device %d: target=%.2f GiB, "
        "reserved %.2f -> %.2f GiB, allocated %.2f -> %.2f GiB, globally free=%.2f/%.2f GiB. "
        "Do not call torch.cuda.empty_cache() while this reservation is needed.",
        device,
        result["target_gib"],
        result["reserved_before_gib"],
        result["reserved_after_gib"],
        result["allocated_before_gib"],
        result["allocated_after_gib"],
        result["free_after_gib"],
        result["total_gib"],
    )
    return result


def load_vipe_configs():
    slam_cfg_default = OmegaConf.load(REPO_ROOT / "configs/slam/default.yaml")
    panorama_cfg = OmegaConf.load(REPO_ROOT / "configs/pipeline/panorama.yaml")
    slam_cfg = OmegaConf.merge(slam_cfg_default, panorama_cfg.slam)
    slam_cfg.visualize = False
    slam_cfg.optimize_intrinsics = False
    slam_cfg.keyframe_depth = None
    return slam_cfg, panorama_cfg.virtual


def load_pvdepth_clips(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as handle:
        clips_raw = json.load(handle)
    if not isinstance(clips_raw, dict):
        raise ValueError(f"{json_path} must contain a JSON object at the top level.")

    clips = []
    for town, town_data in clips_raw.items():
        if not isinstance(town_data, dict):
            raise ValueError(f"Town entry {town!r} must be a JSON object.")
        for source_path, path_data in town_data.items():
            if not isinstance(path_data, dict):
                raise ValueError(f"Path entry {source_path!r} under {town!r} must be a JSON object.")
            for clip_name, frames in path_data.items():
                if not isinstance(frames, list):
                    raise ValueError(f"Frames for clip {clip_name!r} must be a JSON list.")
                clips.append(
                    {
                        "id": clip_name,
                        "frames": frames,
                        "frame_ids": list(range(len(frames))),
                        "source_type": "pvdepth_json",
                    }
                )

    ensure_unique_clip_ids(clips)
    return clips


def load_omniroam_clips(args: argparse.Namespace) -> list[dict]:
    with args.json_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    if not isinstance(split, dict):
        raise ValueError(f"{args.json_path} must contain a JSON object at the top level.")
    video_ids = split.get(args.split_subset)
    if not isinstance(video_ids, list):
        raise ValueError(f"{args.json_path} must contain split[{args.split_subset!r}] as a list.")
    if not video_ids:
        raise ValueError(f"No video IDs found in split {args.split_subset!r}: {args.json_path}")

    clips = []
    for video_id in video_ids:
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"Invalid video_id in split {args.split_subset!r}: {video_id!r}")
        clip = {"id": video_id, "source_type": args.dataset_format}
        if args.dataset_format != "pose_i2v_results":
            clip["frame_ids"] = list(range(1, args.interiorgs_max_frames + 1))
        clips.append(clip)

    ensure_unique_clip_ids(clips)
    return clips


def ensure_unique_clip_ids(clips: list[dict]) -> None:
    clip_name_counts = Counter(clip["id"] for clip in clips)
    duplicate_names = sorted(name for name, count in clip_name_counts.items() if count > 1)
    if duplicate_names:
        raise ValueError(f"Duplicate clip names would overwrite pose outputs: {duplicate_names}")


def validate_pvdepth_clip(clip: dict, image_base_dir: Path) -> dict:
    if not clip["frames"]:
        raise ValueError(f"Clip {clip['id']} contains no frames.")

    image_paths = []
    for frame_idx, frame_info in enumerate(clip["frames"]):
        if not isinstance(frame_info, dict) or not frame_info.get("rgb_path"):
            raise ValueError(f"Clip {clip['id']} frame {frame_idx} has no rgb_path.")
        image_path = image_base_dir / frame_info["rgb_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
            raise ValueError(f"Image cannot be decoded: {image_path}")
        image_paths.append(image_path)
    return {
        "kind": "png_paths",
        "length": len(image_paths),
        "image_paths": image_paths,
    }


def validate_omniroam_png_clip(clip: dict, args: argparse.Namespace) -> dict:
    frame_ext = args.interiorgs_frame_ext.lower().lstrip(".")
    frames_subdir = args.interiorgs_frames_subdir.strip("/")
    image_paths = []
    for frame_id in clip["frame_ids"]:
        image_path = args.image_base_dir / clip["id"] / frames_subdir / f"frame_{frame_id:04d}.{frame_ext}"
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
            raise ValueError(f"Image cannot be decoded: {image_path}")
        image_paths.append(image_path)
    return {
        "kind": "png_paths",
        "length": len(image_paths),
        "image_paths": image_paths,
    }


def validate_omniroam_h5_clip(clip: dict, args: argparse.Namespace) -> dict:
    import h5py

    h5_path = args.h5_data_root / f"{clip['id']}.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(f"OmniRoam H5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as handle:
        for key in ("frame_id", "rgb"):
            if key not in handle:
                raise RuntimeError(f"OmniRoam H5 file is missing dataset {key!r}: {h5_path}")
        frame_ids = np.asarray(handle["frame_id"][()], dtype=np.int64)
        rgb = handle["rgb"]
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise RuntimeError(f"Expected rgb shape [N,H,W,3] in {h5_path}, got {rgb.shape}")
        if rgb.shape[0] != frame_ids.shape[0]:
            raise RuntimeError(
                f"frame_id length and rgb length differ in {h5_path}: {frame_ids.shape[0]} vs {rgb.shape[0]}"
            )

    row_map = {}
    for row_idx, frame_id in enumerate(frame_ids.tolist()):
        frame_id = int(frame_id)
        if frame_id in row_map:
            raise RuntimeError(f"Duplicate frame_id={frame_id} in {h5_path}")
        row_map[frame_id] = row_idx

    rows = []
    for frame_id in clip["frame_ids"]:
        if int(frame_id) not in row_map:
            raise KeyError(f"Missing frame_id={frame_id} in {h5_path}")
        rows.append(row_map[int(frame_id)])

    return {
        "kind": "h5_rgb",
        "length": len(rows),
        "h5_path": h5_path,
        "rows": rows,
    }


def _load_json_object(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _parse_generated_crop(
    segment: dict,
    segment_id: int,
    *,
    format_version: str | None = None,
) -> tuple[int, int, int, int]:
    layout = segment.get("frame_layout")
    if not isinstance(layout, dict):
        raise ValueError(f"segment_{segment_id:02d} is missing frame_layout.")
    crop = layout.get("generated_crop")
    if crop is None and format_version == POSE_RENDER_I2V_MOGE_SEGMENT_FIRST_FORMAT_VERSION:
        width = layout.get("width")
        total_height = layout.get("height")
        dimensions_are_integers = (
            not isinstance(width, bool)
            and isinstance(width, int)
            and not isinstance(total_height, bool)
            and isinstance(total_height, int)
        )
        if dimensions_are_integers:
            content_height = total_height - POSE_I2V_COMPARISON_SEPARATOR_HEIGHT
            if width > 0 and content_height > 0 and content_height % 2 == 0:
                crop = [0, 0, width, content_height // 2]
    if (
        not isinstance(crop, list)
        or len(crop) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in crop)
    ):
        raise ValueError(f"segment_{segment_id:02d} has invalid frame_layout.generated_crop={crop!r}.")
    left, top, right, bottom = crop
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError(f"segment_{segment_id:02d} has invalid generated_crop={crop!r}.")
    return left, top, right, bottom


def crop_comparison_image(
    image: np.ndarray,
    crop: tuple[int, int, int, int],
    *,
    image_path: Path | None = None,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        location = f" for {image_path}" if image_path is not None else ""
        raise ValueError(f"Expected an HxWx3 comparison image{location}, got shape {image.shape}.")
    left, top, right, bottom = crop
    height, width = image.shape[:2]
    if left < 0 or top < 0 or right > width or bottom > height or right <= left or bottom <= top:
        location = f" in {image_path}" if image_path is not None else ""
        raise ValueError(f"generated_crop={list(crop)} is outside image size {width}x{height}{location}.")
    return image[top:bottom, left:right]


def validate_pose_i2v_clip(clip: dict, args: argparse.Namespace) -> dict:
    scene_root = args.pose_i2v_results_root / clip["id"]
    metadata_path = scene_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Scene metadata not found: {metadata_path}")
    metadata = _load_json_object(metadata_path)
    if metadata.get("scene_id") != clip["id"]:
        raise ValueError(
            f"Scene metadata id mismatch in {metadata_path}: expected {clip['id']!r}, "
            f"got {metadata.get('scene_id')!r}."
        )
    segments = metadata.get("segments")
    if segments is None:
        refine = metadata.get("refine")
        if isinstance(refine, dict):
            segments = refine.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(
            f"{metadata_path} must contain a non-empty segments list either at the top level "
            "or under refine."
        )

    if args.pose_i2v_scope == "first_segment":
        selected_segments = segments[:1]
        expected_segment_ids = [0]
    else:
        if len(segments) != args.pose_i2v_expected_segments:
            raise ValueError(
                f"{metadata_path} has {len(segments)} segments; expected exactly "
                f"{args.pose_i2v_expected_segments}."
            )
        selected_segments = segments
        expected_segment_ids = list(range(args.pose_i2v_expected_segments))

    records = []
    first_crop = None
    crops_are_identical = True
    segment_crops = {}
    common_crop_size = None
    previous_frame_id = None
    for expected_segment_id, segment in zip(expected_segment_ids, selected_segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Segment entry {expected_segment_id} in {metadata_path} must be an object.")
        segment_id = segment.get("segment_id")
        if segment_id != expected_segment_id:
            raise ValueError(
                f"Expected segment_id={expected_segment_id} in {metadata_path}, found {segment_id!r}."
            )
        format_version = segment.get("output_format_version") or metadata.get("output_format_version")
        crop = _parse_generated_crop(segment, segment_id, format_version=format_version)
        segment_crops[segment_id] = crop
        if first_crop is None:
            first_crop = crop
        elif crop != first_crop:
            crops_are_identical = False

        saved_ids = segment.get("saved_frame_indices")
        if not isinstance(saved_ids, list) or not saved_ids:
            raise ValueError(f"segment_{segment_id:02d} is missing non-empty saved_frame_indices.")
        for value in saved_ids:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"segment_{segment_id:02d} has invalid saved frame id {value!r}.")
            frame_id = int(value)
            if previous_frame_id is not None and frame_id <= previous_frame_id:
                raise ValueError(
                    f"saved_frame_indices must be globally strictly increasing; got frame_id={frame_id} "
                    f"after {previous_frame_id}."
                )
            image_path = scene_root / f"segment_{segment_id:02d}" / "frames" / f"frame_{frame_id:04d}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"Comparison image not found: {image_path}")
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Comparison image cannot be decoded: {image_path}")
            cropped = crop_comparison_image(image_bgr, crop, image_path=image_path)
            crop_size = cropped.shape[:2]
            if crop_size[1] != crop_size[0] * 2:
                raise ValueError(
                    f"Expected a 2:1 ERP crop in {image_path}, got {crop_size[1]}x{crop_size[0]}."
                )
            if common_crop_size is None:
                common_crop_size = crop_size
            elif crop_size != common_crop_size:
                raise ValueError(
                    f"Crop size changed in {image_path}: expected {common_crop_size[1]}x{common_crop_size[0]}, "
                    f"got {crop_size[1]}x{crop_size[0]}."
                )
            records.append(
                {
                    "frame_id": frame_id,
                    "segment_id": segment_id,
                    "image_path": image_path,
                    "generated_crop": crop,
                }
            )
            previous_frame_id = frame_id

    if not records or first_crop is None or common_crop_size is None:
        raise ValueError(f"No pose-I2V frames selected for scene {clip['id']}.")
    return {
        "kind": "comparison_png",
        "length": len(records),
        "records": records,
        "frame_ids": [record["frame_id"] for record in records],
        "generated_crop": first_crop if crops_are_identical else None,
        "segment_generated_crops": segment_crops,
        "frame_size": common_crop_size,
        "metadata_path": metadata_path,
        "source_format_version": metadata.get("output_format_version"),
        "input_root": args.pose_i2v_results_root,
        "scope": args.pose_i2v_scope,
    }


def validate_clip_source(clip: dict, args: argparse.Namespace) -> dict:
    if args.dataset_format == "pvdepth_json":
        return validate_pvdepth_clip(clip, args.image_base_dir)
    if args.dataset_format == "omniroam_png":
        return validate_omniroam_png_clip(clip, args)
    if args.dataset_format == "omniroam_h5":
        return validate_omniroam_h5_clip(clip, args)
    if args.dataset_format == "pose_i2v_results":
        return validate_pose_i2v_clip(clip, args)
    raise ValueError(f"Unsupported dataset_format: {args.dataset_format}")


def create_video_stream(source: dict, resolution: tuple[int, int] | None, name: str):
    from vipe.streams.base import CameraType, VideoFrame, VideoStream

    class PoseVideoStream(VideoStream):
        def __init__(self):
            if resolution is None:
                self.height, self.width = source["frame_size"]
            else:
                self.width, self.height = resolution

        def frame_size(self) -> tuple[int, int]:
            return self.height, self.width

        def fps(self) -> float:
            return 10.0

        def __len__(self) -> int:
            return source["length"]

        def name(self) -> str:
            return name

        def __iter__(self):
            if source["kind"] == "png_paths":
                iterator = self._iter_png_paths()
            elif source["kind"] == "h5_rgb":
                iterator = self._iter_h5_rgb()
            elif source["kind"] == "comparison_png":
                iterator = self._iter_comparison_png()
            else:
                raise ValueError(f"Unsupported frame source kind: {source['kind']}")
            yield from iterator

        def _iter_png_paths(self):
            for frame_idx, image_path in enumerate(source["image_paths"]):
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RuntimeError(f"Failed to read validated image: {image_path}")
                image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                yield self._make_frame(frame_idx, image)

        def _iter_h5_rgb(self):
            import h5py

            with h5py.File(source["h5_path"], "r") as handle:
                rgb = handle["rgb"]
                for frame_idx, row in enumerate(source["rows"]):
                    image = np.asarray(rgb[row], dtype=np.uint8)
                    if image.ndim != 3 or image.shape[-1] != 3:
                        raise RuntimeError(
                            f"Expected H5 RGB row shape [H,W,3] at row {row}, got {image.shape}"
                        )
                    yield self._make_frame(frame_idx, image)

        def _iter_comparison_png(self):
            for frame_idx, record in enumerate(source["records"]):
                image_path = record["image_path"]
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RuntimeError(f"Failed to read validated comparison image: {image_path}")
                image_bgr = crop_comparison_image(
                    image_bgr,
                    record["generated_crop"],
                    image_path=image_path,
                )
                image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                yield self._make_frame(frame_idx, image)

        def _make_frame(self, frame_idx: int, image: np.ndarray):
            if image.shape[:2] != (self.height, self.width):
                if source["kind"] == "comparison_png":
                    raise RuntimeError(
                        f"Native comparison crop size changed: expected {self.width}x{self.height}, "
                        f"got {image.shape[1]}x{image.shape[0]}."
                    )
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
            return VideoFrame(
                raw_frame_idx=frame_idx,
                rgb=torch.from_numpy(image).float() / 255.0,
                camera_type=CameraType.PANORAMA,
            )

    return PoseVideoStream()


def compute_virtual_view_size(virtual_cfg, virtual_view_height: int) -> tuple[int, int, float]:
    if virtual_view_height <= 0:
        raise ValueError("virtual_view_height must be positive.")
    virtual_height = virtual_view_height + virtual_view_height % 2
    virtual_focal = virtual_height / (2 * np.tan(np.deg2rad(virtual_cfg.fovx) / 2))
    virtual_width = int(virtual_focal * np.tan(np.deg2rad(virtual_cfg.fovx) / 2) * 2)
    virtual_width += virtual_width % 2
    return virtual_height, virtual_width, float(virtual_focal)


def compute_standard_slam_size(frame_size: tuple[int, int]) -> tuple[int, int]:
    height, width = frame_size
    scale_factor = np.sqrt((384 * 512) / (height * width))
    scaled_height = int(height * scale_factor)
    scaled_width = int(width * scale_factor)
    return scaled_height - scaled_height % 8, scaled_width - scaled_width % 8


def build_slam_streams(cached_video_stream, virtual_cfg, virtual_view_height: int):
    from vipe.ext import lietorch as lt
    from vipe.pipeline.processors import EquirectProjectionProcessor
    from vipe.streams.base import ProcessedVideoStream
    from vipe.utils.geometry import se3_to_so3, so3_to_se3

    virtual_height, virtual_width, virtual_focal = compute_virtual_view_size(virtual_cfg, virtual_view_height)

    virtual_intrinsics = torch.tensor(
        [virtual_focal, virtual_focal, virtual_width // 2, virtual_height // 2],
        dtype=torch.float32,
        device="cuda",
    )
    virtual_size = virtual_height, virtual_width

    rig_transforms = [
        so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(yaw, 0.0))
        for yaw in np.linspace(0, 2 * np.pi, virtual_cfg.num_views, endpoint=False)
    ]
    if virtual_cfg.top:
        rig_transforms.append(so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(0.0, np.pi / 2)))
    if virtual_cfg.bottom:
        rig_transforms.append(so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(0.0, -np.pi / 2)))

    slam_streams = []
    for rig_transform in rig_transforms:
        projector = EquirectProjectionProcessor(se3_to_so3(rig_transform), virtual_size, virtual_intrinsics)
        slam_streams.append(ProcessedVideoStream(cached_video_stream, [projector]).cache(online=True))
    return slam_streams, lt.stack(rig_transforms, dim=0), virtual_size


def save_poses_atomically(pose_path: Path, poses: np.ndarray) -> None:
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=pose_path.parent,
            prefix=f".{pose_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(poses.tolist(), handle, indent=2)
            handle.write("\n")
        os.replace(temporary_path, pose_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_json_atomically(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_pose_i2v_manifest(clip: dict, source: dict, virtual_cfg, virtual_size: tuple[int, int]) -> dict:
    input_root = source["input_root"]
    metadata_path = source["metadata_path"]
    frames = []
    for record in source["records"]:
        frames.append(
            {
                "frame_id": record["frame_id"],
                "segment_id": record["segment_id"],
                "comparison_png": str(record["image_path"].relative_to(input_root)),
            }
        )
    return {
        "schema_version": "vipe_pose_i2v_input_manifest_v1",
        "scene_id": clip["id"],
        "scope": source["scope"],
        "input_root": str(input_root.resolve()),
        "metadata_path": str(metadata_path.relative_to(input_root)),
        "source_output_format_version": source["source_format_version"],
        "frame_count": source["length"],
        "frames": frames,
        "generated_crop": list(source["generated_crop"]) if source["generated_crop"] is not None else None,
        "segment_generated_crops": {
            str(segment_id): list(crop) for segment_id, crop in source["segment_generated_crops"].items()
        },
        "crop_size_hw": list(source["frame_size"]),
        "input_processing": {
            "crop_coordinates": "half-open [left, top, right, bottom]",
            "resize_before_projection": False,
        },
        "virtual_view_size_hw": list(virtual_size),
        "slam_input_size_hw": list(compute_standard_slam_size(virtual_size)),
        "virtual_camera": {
            "fovx_degrees": float(virtual_cfg.fovx),
            "horizontal_views": int(virtual_cfg.num_views),
            "top": bool(virtual_cfg.top),
            "bottom": bool(virtual_cfg.bottom),
        },
    }


def existing_pose_i2v_result_is_valid(
    pose_path: Path,
    manifest_path: Path,
    *,
    scene_id: str,
    scope: str,
    virtual_size: tuple[int, int],
) -> bool:
    try:
        poses = json.loads(pose_path.read_text(encoding="utf-8"))
        manifest = _load_json_object(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    frames = manifest.get("frames")
    if (
        manifest.get("schema_version") != "vipe_pose_i2v_input_manifest_v1"
        or manifest.get("scene_id") != scene_id
        or manifest.get("scope") != scope
        or manifest.get("virtual_view_size_hw") != list(virtual_size)
        or not isinstance(frames, list)
        or not isinstance(poses, list)
        or len(frames) != len(poses)
        or manifest.get("frame_count") != len(frames)
    ):
        return False
    if any(
        not isinstance(pose, list)
        or len(pose) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in pose)
        for pose in poses
    ):
        return False
    frame_ids = [frame.get("frame_id") for frame in frames if isinstance(frame, dict)]
    return len(frame_ids) == len(frames) and all(
        isinstance(frame_id, int) and (idx == 0 or frame_id > frame_ids[idx - 1])
        for idx, frame_id in enumerate(frame_ids)
    )


def estimate_clip_pose(clip: dict, source: dict, pose_path: Path, args, slam_cfg, virtual_cfg) -> tuple[int, int]:
    from vipe.slam.system import SLAMSystem
    from vipe.streams.base import CachedVideoStream

    video_stream = None
    cached_video_stream = None
    slam_streams = []
    slam_pipeline = None
    slam_output = None
    try:
        resolution = None if source["kind"] == "comparison_png" else (args.resolution, args.resolution // 2)
        video_stream = create_video_stream(source, resolution=resolution, name=clip["id"])
        cached_video_stream = CachedVideoStream(video_stream, desc=f"Loading {clip['id']}")
        virtual_view_height = args.virtual_view_height
        if virtual_view_height is None:
            virtual_view_height = 256 if source["kind"] == "comparison_png" else args.resolution // 4
        slam_streams, rig_se3, virtual_size = build_slam_streams(
            cached_video_stream,
            virtual_cfg,
            virtual_view_height,
        )

        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=slam_cfg)
        slam_output = slam_pipeline.run(slam_streams, rig=rig_se3)
        poses = slam_output.trajectory.matrix().detach().cpu().numpy()
        expected_shape = (source["length"], 4, 4)
        if poses.shape != expected_shape:
            raise RuntimeError(f"Expected pose shape {expected_shape}, got {poses.shape}.")
        if not np.isfinite(poses).all():
            raise RuntimeError("Estimated poses contain NaN or Inf values.")
        save_poses_atomically(pose_path, poses)
        return virtual_size
    finally:
        del slam_output, slam_pipeline, slam_streams, cached_video_stream, video_stream
        gc.collect()


def main() -> int:
    args = parse_args()
    if args.resolution <= 0 or args.resolution % 2 != 0:
        raise ValueError("--resolution must be a positive even integer.")
    if args.interiorgs_max_frames <= 0:
        raise ValueError("--interiorgs_max_frames must be positive.")
    if args.pose_i2v_expected_segments <= 0:
        raise ValueError("--pose_i2v_expected_segments must be positive.")
    if args.virtual_view_height is not None and args.virtual_view_height <= 0:
        raise ValueError("--virtual_view_height must be positive.")
    if args.start_clip_idx < 0:
        raise ValueError("--start_clip_idx must be non-negative.")
    if args.num_clips < 0:
        raise ValueError("--num_clips must be non-negative.")
    if not math.isfinite(args.cuda_reserve_gib) or args.cuda_reserve_gib < 0:
        raise ValueError("--cuda_reserve_gib must be finite and non-negative.")
    if not math.isfinite(args.cuda_reserve_safety_gib) or args.cuda_reserve_safety_gib < 0:
        raise ValueError("--cuda_reserve_safety_gib must be finite and non-negative.")
    if not args.json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {args.json_path}")
    if args.dataset_format in {"pvdepth_json", "omniroam_png"}:
        if args.image_base_dir is None or not args.image_base_dir.is_dir():
            raise FileNotFoundError(f"Image base directory not found: {args.image_base_dir}")
    if args.dataset_format == "omniroam_h5":
        if args.h5_data_root is None or not args.h5_data_root.is_dir():
            raise FileNotFoundError(f"H5 data root not found: {args.h5_data_root}")
    if args.dataset_format == "pose_i2v_results":
        if args.pose_i2v_results_root is None or not args.pose_i2v_results_root.is_dir():
            raise FileNotFoundError(f"Pose-I2V results root not found: {args.pose_i2v_results_root}")

    if args.dataset_format == "pvdepth_json":
        clips = load_pvdepth_clips(args.json_path)
        output_dir = args.output_root_dir / args.json_path.stem
    elif args.dataset_format in {"omniroam_png", "omniroam_h5"}:
        clips = load_omniroam_clips(args)
        output_dir = args.output_root_dir / f"{args.json_path.stem}_{args.dataset_format}"
    else:
        clips = load_omniroam_clips(args)
        output_dir = args.output_root_dir
    slam_cfg, virtual_cfg = load_vipe_configs()
    output_dir.mkdir(parents=True, exist_ok=True)

    succeeded = []
    skipped = []
    failed = []
    total_clips = len(clips)
    if args.num_clips > 0:
        selected_clips = clips[args.start_clip_idx : args.start_clip_idx + args.num_clips]
    else:
        selected_clips = clips[args.start_clip_idx :]
    logger.info(
        "Loaded %d clips from %s; selected %d clips from start_clip_idx=%d num_clips=%d",
        total_clips,
        args.json_path,
        len(selected_clips),
        args.start_clip_idx,
        args.num_clips,
    )
    if selected_clips and args.cuda_reserve_gib > 0:
        reserve_cuda_memory(args.cuda_reserve_gib, args.cuda_reserve_safety_gib)

    for clip_idx, clip in enumerate(selected_clips, start=1):
        clip_name = clip["id"]
        if args.dataset_format == "pose_i2v_results":
            pose_path = output_dir / "poses" / f"{clip_name}_poses.json"
            manifest_path = output_dir / "poses" / f"{clip_name}_input_manifest.json"
        else:
            pose_path = output_dir / f"{clip_name}_poses.json"
            manifest_path = None
        logger.info("[%d/%d] Processing %s", clip_idx, len(selected_clips), clip_name)

        if pose_path.exists() and not args.overwrite:
            requested_virtual_height = args.virtual_view_height
            if requested_virtual_height is None:
                requested_virtual_height = 256 if args.dataset_format == "pose_i2v_results" else args.resolution // 4
            expected_virtual_size = compute_virtual_view_size(virtual_cfg, requested_virtual_height)[:2]
            if manifest_path is None or existing_pose_i2v_result_is_valid(
                pose_path,
                manifest_path,
                scene_id=clip_name,
                scope=args.pose_i2v_scope,
                virtual_size=expected_virtual_size,
            ):
                logger.info("Skipping existing result: %s", pose_path)
                skipped.append(clip_name)
                continue
            logger.warning("Existing pose has no valid matching manifest; recomputing: %s", pose_path)

        start_time = time.time()
        try:
            source = validate_clip_source(clip, args)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required by the ViPE panorama SLAM pipeline.")
            virtual_size = estimate_clip_pose(clip, source, pose_path, args, slam_cfg, virtual_cfg)
            if manifest_path is not None:
                manifest = build_pose_i2v_manifest(clip, source, virtual_cfg, virtual_size)
                save_json_atomically(manifest_path, manifest)
            succeeded.append(clip_name)
            logger.info("Saved %s in %.2f seconds", pose_path, time.time() - start_time)
        except Exception as error:
            failed.append((clip_name, str(error)))
            logger.exception("Failed clip %s", clip_name)

    logger.info(
        "Summary: succeeded=%d, skipped=%d, failed=%d",
        len(succeeded),
        len(skipped),
        len(failed),
    )
    for clip_name, error in failed:
        logger.error("Failed: %s: %s", clip_name, error)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
