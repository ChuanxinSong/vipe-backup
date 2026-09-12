import argparse
import gc
import json
import logging
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate and save ViPE poses for panorama image sequences.")
    parser.add_argument("--json_path", type=Path, required=True, help="Input JSON describing panorama clips.")
    parser.add_argument(
        "--dataset_format",
        choices=["pvdepth_json", "omniroam_png", "omniroam_h5"],
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
    parser.add_argument("--start_clip_idx", type=int, default=0, help="First loaded clip index to process.")
    parser.add_argument("--num_clips", type=int, default=0, help="Maximum clips to process; 0 means all remaining clips.")
    parser.add_argument("--overwrite", action="store_true", help="Re-estimate clips whose pose output already exists.")
    return parser.parse_args()


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

    frame_ids = list(range(1, args.interiorgs_max_frames + 1))
    clips = []
    for video_id in video_ids:
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"Invalid video_id in split {args.split_subset!r}: {video_id!r}")
        clips.append(
            {
                "id": video_id,
                "frame_ids": frame_ids,
                "source_type": args.dataset_format,
            }
        )

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


def validate_clip_source(clip: dict, args: argparse.Namespace) -> dict:
    if args.dataset_format == "pvdepth_json":
        return validate_pvdepth_clip(clip, args.image_base_dir)
    if args.dataset_format == "omniroam_png":
        return validate_omniroam_png_clip(clip, args)
    if args.dataset_format == "omniroam_h5":
        return validate_omniroam_h5_clip(clip, args)
    raise ValueError(f"Unsupported dataset_format: {args.dataset_format}")


def create_video_stream(source: dict, resolution: tuple[int, int], name: str):
    from vipe.streams.base import CameraType, VideoFrame, VideoStream

    class PoseVideoStream(VideoStream):
        def __init__(self):
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

        def _make_frame(self, frame_idx: int, image: np.ndarray):
            if image.shape[:2] != (self.height, self.width):
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
            return VideoFrame(
                raw_frame_idx=frame_idx,
                rgb=torch.from_numpy(image).float() / 255.0,
                camera_type=CameraType.PANORAMA,
            )

    return PoseVideoStream()


def build_slam_streams(cached_video_stream, virtual_cfg, resolution: int):
    from vipe.ext import lietorch as lt
    from vipe.pipeline.processors import EquirectProjectionProcessor
    from vipe.streams.base import ProcessedVideoStream
    from vipe.utils.geometry import se3_to_so3, so3_to_se3

    panorama_height = resolution // 2
    virtual_height = panorama_height // 2
    virtual_height += virtual_height % 2
    virtual_focal = virtual_height / (2 * np.tan(np.deg2rad(virtual_cfg.fovx) / 2))
    virtual_width = int(virtual_focal * np.tan(np.deg2rad(virtual_cfg.fovx) / 2) * 2)
    virtual_width += virtual_width % 2

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
    return slam_streams, lt.stack(rig_transforms, dim=0)


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


def estimate_clip_pose(clip: dict, source: dict, pose_path: Path, args, slam_cfg, virtual_cfg) -> None:
    from vipe.slam.system import SLAMSystem
    from vipe.streams.base import CachedVideoStream

    video_stream = None
    cached_video_stream = None
    slam_streams = []
    slam_pipeline = None
    slam_output = None
    try:
        video_stream = create_video_stream(
            source,
            resolution=(args.resolution, args.resolution // 2),
            name=clip["id"],
        )
        cached_video_stream = CachedVideoStream(video_stream, desc=f"Loading {clip['id']}")
        slam_streams, rig_se3 = build_slam_streams(cached_video_stream, virtual_cfg, args.resolution)

        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=slam_cfg)
        slam_output = slam_pipeline.run(slam_streams, rig=rig_se3)
        poses = slam_output.trajectory.matrix().detach().cpu().numpy()
        expected_shape = (source["length"], 4, 4)
        if poses.shape != expected_shape:
            raise RuntimeError(f"Expected pose shape {expected_shape}, got {poses.shape}.")
        if not np.isfinite(poses).all():
            raise RuntimeError("Estimated poses contain NaN or Inf values.")
        save_poses_atomically(pose_path, poses)
    finally:
        del slam_output, slam_pipeline, slam_streams, cached_video_stream, video_stream
        gc.collect()


def main() -> int:
    args = parse_args()
    if args.resolution <= 0 or args.resolution % 2 != 0:
        raise ValueError("--resolution must be a positive even integer.")
    if args.interiorgs_max_frames <= 0:
        raise ValueError("--interiorgs_max_frames must be positive.")
    if args.start_clip_idx < 0:
        raise ValueError("--start_clip_idx must be non-negative.")
    if args.num_clips < 0:
        raise ValueError("--num_clips must be non-negative.")
    if not args.json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {args.json_path}")
    if args.dataset_format in {"pvdepth_json", "omniroam_png"}:
        if args.image_base_dir is None or not args.image_base_dir.is_dir():
            raise FileNotFoundError(f"Image base directory not found: {args.image_base_dir}")
    if args.dataset_format == "omniroam_h5":
        if args.h5_data_root is None or not args.h5_data_root.is_dir():
            raise FileNotFoundError(f"H5 data root not found: {args.h5_data_root}")

    if args.dataset_format == "pvdepth_json":
        clips = load_pvdepth_clips(args.json_path)
        output_dir = args.output_root_dir / args.json_path.stem
    else:
        clips = load_omniroam_clips(args)
        output_dir = args.output_root_dir / f"{args.json_path.stem}_{args.dataset_format}"
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

    for clip_idx, clip in enumerate(selected_clips, start=1):
        clip_name = clip["id"]
        pose_path = output_dir / f"{clip_name}_poses.json"
        logger.info("[%d/%d] Processing %s", clip_idx, len(selected_clips), clip_name)

        if pose_path.exists() and not args.overwrite:
            logger.info("Skipping existing result: %s", pose_path)
            skipped.append(clip_name)
            continue

        start_time = time.time()
        try:
            source = validate_clip_source(clip, args)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required by the ViPE panorama SLAM pipeline.")
            estimate_clip_pose(clip, source, pose_path, args, slam_cfg, virtual_cfg)
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
