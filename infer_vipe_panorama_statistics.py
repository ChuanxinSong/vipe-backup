import argparse
import gc
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from vipe.ext import lietorch as lt
from vipe.pipeline.panorama import MergedPanoramaVideoStream
from vipe.pipeline.processors import EquirectProjectionProcessor
from vipe.slam.system import SLAMSystem
from vipe.streams.base import CachedVideoStream, CameraType, ProcessedVideoStream, VideoFrame, VideoStream
from vipe.utils.geometry import se3_to_so3, so3_to_se3


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infer_vipe")

PROFILE_NONE = "none"
PROFILE_MEMORY = "memory"
PROFILE_FLOPS = "flops"
PROFILE_BOTH = "both"
PROFILE_SCOPE = "full_vipe_pipeline"


try:
    from depth_eval.utils import save_video, vis_sequence_depth
except ImportError:
    print("Warning: depth_eval.utils not found. Using fallback visualization.")

    def vis_sequence_depth(depths):
        vis = []
        for depth in depths:
            disparity = 1.0 / (depth + 1e-6)
            disparity_min, disparity_max = disparity.min(), disparity.max()
            disparity_norm = (disparity - disparity_min) / (disparity_max - disparity_min + 1e-8)
            disparity_color = cv2.applyColorMap((disparity_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            vis.append(cv2.cvtColor(disparity_color, cv2.COLOR_BGR2RGB) / 255.0)
        return np.array(vis)

    def save_video(frames, path, fps=20):
        if len(frames) == 0:
            return
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame in frames:
            out.write((frame * 255).astype(np.uint8)[:, :, ::-1])
        out.release()


class SimpleJsonVideoStream(VideoStream):
    def __init__(self, image_paths, resolution=(1024, 512), fps=10.0, name="json_stream"):
        self.image_paths = image_paths
        self.width = resolution[0]
        self.height = resolution[1]
        self._fps = fps
        self._name = name

    def frame_size(self) -> tuple[int, int]:
        return (self.height, self.width)

    def fps(self) -> float:
        return self._fps

    def __len__(self) -> int:
        return len(self.image_paths)

    def name(self) -> str:
        return self._name

    def __iter__(self):
        for frame_idx, path in enumerate(self.image_paths):
            if not os.path.exists(path):
                logger.error(f"Image not found: {path}")
                image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                image = cv2.imread(path)
                if image is None:
                    logger.error(f"Failed to load image: {path}")
                    image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                elif image.shape[1] != self.width or image.shape[0] != self.height:
                    image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = torch.from_numpy(image).float() / 255.0
            yield VideoFrame(raw_frame_idx=frame_idx, rgb=rgb, camera_type=CameraType.PANORAMA)


def extract_file_parameters(json_filename):
    setting = "dynamic" if "dynamic" in json_filename else "static"
    fps_str = next((x for x in ["fps01", "fps10", "fps20", "fps02"] if x in json_filename), "fps_unknown")
    len_str = next((x for x in ["len20", "len50", "len90", "len110"] if x in json_filename), "len_unknown")
    return setting, fps_str, len_str


def load_vipe_configs():
    slam_cfg_default = OmegaConf.load("configs/slam/default.yaml")
    pano_pipeline_cfg = OmegaConf.load("configs/pipeline/panorama.yaml")
    if "slam" in pano_pipeline_cfg:
        slam_cfg = OmegaConf.merge(slam_cfg_default, pano_pipeline_cfg.slam)
    else:
        slam_cfg = slam_cfg_default

    slam_cfg.visualize = False
    if OmegaConf.is_missing(slam_cfg, "optimize_intrinsics"):
        slam_cfg.optimize_intrinsics = False
    slam_cfg.keyframe_depth = None

    return slam_cfg, pano_pipeline_cfg.virtual


def flatten_clips(clips_raw):
    all_clips = []
    for town, town_data in clips_raw.items():
        for path, path_data in town_data.items():
            for clip_name, frames in path_data.items():
                all_clips.append(
                    {
                        "town": town,
                        "path": path,
                        "id": clip_name,
                        "frames": frames,
                    }
                )
    return all_clips


def maybe_sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profile_enabled(args):
    return args.profile_mode != PROFILE_NONE


def flops_enabled(args):
    return args.profile_mode in {PROFILE_FLOPS, PROFILE_BOTH}


def format_flops_human(flops_value):
    if flops_value >= 1e12:
        return f"{flops_value / 1e12:.3f} TFLOPs"
    return f"{flops_value / 1e9:.3f} GFLOPs"


def format_memory_gib(memory_bytes):
    return memory_bytes / (1024 ** 3)


def build_output_paths(args, clip_name):
    setting, fps_str, len_str = extract_file_parameters(args.json_path)
    out_dir = os.path.join(
        args.output_root_dir,
        f"town0210_{args.resolution}_{setting}_{fps_str}_{len_str}_{args.depth_method}",
    )
    npy_path = os.path.join(out_dir, f"{clip_name}_distance.npy")
    video_path = os.path.join(out_dir, f"{clip_name}_vis.mp4")
    return out_dir, npy_path, video_path


def build_depth_options(args, clip_name):
    depth_options = {
        "unet_path": args.unet_path,
        "ppl_type": args.ppl_type,
        "num_denoising_steps": args.num_denoising_steps,
        "guidance_scale": args.guidance_scale,
        "cpu_offload": args.cpu_offload,
    }

    if args.depthcrafter_npy_dir and args.depth_method == "depthcrafter":
        setting, fps_str, len_str = extract_file_parameters(args.json_path)
        sub_dir = f"town0210_{args.resolution}_{setting}_{fps_str}_{len_str}"
        source_npy = os.path.join(args.depthcrafter_npy_dir, sub_dir, f"{clip_name}_disparity.npy")
        depth_options["npy_path"] = source_npy

    return depth_options


def prepare_clip_paths(clip, args):
    image_paths = []
    for frame_info in clip["frames"]:
        relative_path = frame_info.get("rgb_path")
        if not relative_path:
            raise ValueError(f"Clip {clip['id']} contains a frame without 'rgb_path'.")
        full_path = os.path.join(args.image_base_dir, relative_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        image_paths.append(full_path)
    return image_paths


def execute_vipe_clip(device, clip, slam_cfg_default, virtual_cfg_default, args, collect_visualization):
    clip_name = clip["id"]
    image_paths = prepare_clip_paths(clip, args)
    video_stream = None
    cached_video_stream = None
    slam_pipeline = None
    slam_output = None
    output_stream = None
    slam_streams = []

    try:
        video_stream = SimpleJsonVideoStream(
            image_paths,
            resolution=(args.resolution, args.resolution // 2),
            name=clip_name,
        )
        cached_video_stream = CachedVideoStream(video_stream, desc="Loading Frames")

        virtual_cfg = virtual_cfg_default
        pano_height = args.resolution // 2
        virtual_height = pano_height // 2
        virtual_height = virtual_height + (virtual_height % 2)
        virtual_focal = virtual_height / (2 * np.tan(np.deg2rad(virtual_cfg.fovx) / 2))
        virtual_width = int(virtual_focal * np.tan(np.deg2rad(virtual_cfg.fovx) / 2) * 2)
        virtual_width = virtual_width + (virtual_width % 2)

        virtual_intrinsics = torch.tensor(
            [virtual_focal, virtual_focal, virtual_width // 2, virtual_height // 2],
            dtype=torch.float32,
            device=device,
        )
        virtual_size = (virtual_height, virtual_width)

        rig_transforms = [
            so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(yaw, 0.0))
            for yaw in np.linspace(0, 2 * np.pi, virtual_cfg.num_views, endpoint=False)
        ]
        if virtual_cfg.top:
            rig_transforms.append(so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(0.0, np.pi / 2)))
        if virtual_cfg.bottom:
            rig_transforms.append(
                so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(0.0, -np.pi / 2))
            )

        for rig_transform in rig_transforms:
            projectors = [
                EquirectProjectionProcessor(
                    se3_to_so3(rig_transform),
                    virtual_size,
                    virtual_intrinsics,
                )
            ]
            slam_streams.append(ProcessedVideoStream(cached_video_stream, projectors).cache(online=True))

        rig_se3 = lt.stack(rig_transforms, dim=0)

        logger.info("Running SLAM for clip %s", clip_name)
        slam_pipeline = SLAMSystem(device=device, config=slam_cfg_default)
        slam_output = slam_pipeline.run(slam_streams, rig=rig_se3)

        logger.info("Merging panorama depth for clip %s with %s", clip_name, args.depth_method)
        output_stream = MergedPanoramaVideoStream(
            cached_video_stream,
            slam_output,
            pano_depth_method=args.depth_method,
            depth_options=build_depth_options(args, clip_name),
        )

        all_distances = []
        original_rgbs = [] if collect_visualization else None
        for frame in tqdm(output_stream, total=len(video_stream), desc=f"Fusing {clip_name}"):
            all_distances.append(frame.metric_depth.cpu().numpy())
            if collect_visualization:
                rgb = frame.rgb.cpu().numpy()
                original_rgbs.append((rgb * 255).astype(np.uint8))

        if not all_distances:
            raise RuntimeError(f"Clip {clip_name} produced no frames during fusion.")

        return np.stack(all_distances), original_rgbs
    finally:
        del output_stream, slam_output, slam_pipeline, cached_video_stream, slam_streams, video_stream
        gc.collect()


def profile_clip(device, clip, slam_cfg_default, virtual_cfg_default, args):
    if device.type == "cuda":
        # torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    maybe_sync_cuda()

    start_time = time.time()
    total_flops = None

    if flops_enabled(args):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.inference_mode():
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
            ) as prof:
                all_distances, _ = execute_vipe_clip(
                    device=device,
                    clip=clip,
                    slam_cfg_default=slam_cfg_default,
                    virtual_cfg_default=virtual_cfg_default,
                    args=args,
                    collect_visualization=False,
                )

        total_flops = float(sum(getattr(event, "flops", 0) or 0 for event in prof.key_averages()))
    else:
        with torch.inference_mode():
            all_distances, _ = execute_vipe_clip(
                device=device,
                clip=clip,
                slam_cfg_default=slam_cfg_default,
                virtual_cfg_default=virtual_cfg_default,
                args=args,
                collect_visualization=False,
            )

    maybe_sync_cuda()
    wall_time_sec = time.time() - start_time

    peak_allocated = 0
    peak_reserved = 0
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())

    result = {
        "clip_name": clip["id"],
        "town": clip["town"],
        "path": clip["path"],
        "num_frames": int(all_distances.shape[0]),
        "input_resolution": {
            "width": int(args.resolution),
            "height": int(args.resolution // 2),
        },
        "wall_time_sec": float(wall_time_sec),
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "peak_memory_allocated_gib": format_memory_gib(peak_allocated),
        "peak_memory_reserved_gib": format_memory_gib(peak_reserved),
    }

    if total_flops is not None:
        frames = max(int(all_distances.shape[0]), 1)
        result.update(
            {
                "total_flops": total_flops,
                "total_flops_human": format_flops_human(total_flops),
                "flops_per_frame": total_flops / frames,
                "flops_per_frame_human": format_flops_human(total_flops / frames),
            }
        )

    del all_distances
    gc.collect()
    if device.type == "cuda":
        # torch.cuda.empty_cache()
        pass

    return result


def save_clip_outputs(all_distances, original_rgbs, npy_path, video_path):
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, all_distances)
    print(f" -> Saved NPY: {npy_path}")

    print(" -> Generating video...")
    disparity = 1.0 / (all_distances + 1e-6)
    disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)
    vis_colored = vis_sequence_depth(disparity)

    combined_frames = []
    for idx in range(len(original_rgbs)):
        rgb_float = original_rgbs[idx].astype(np.float32) / 255.0
        depth_vis = vis_colored[idx]
        combined_frames.append(np.concatenate([rgb_float, depth_vis], axis=0))

    save_video(combined_frames, video_path, fps=10)
    print(f" -> Video saved: {video_path}")


def make_profile_summary(results):
    if not results:
        return {
            "successful_clips": 0,
            "average_wall_time_sec": None,
            "average_peak_memory_allocated_gib": None,
            "average_peak_memory_reserved_gib": None,
            "average_total_flops": None,
            "average_total_flops_human": None,
            "average_flops_per_frame": None,
            "average_flops_per_frame_human": None,
        }

    summary = {
        "successful_clips": len(results),
        "average_wall_time_sec": float(np.mean([item["wall_time_sec"] for item in results])),
        "average_peak_memory_allocated_gib": float(np.mean([item["peak_memory_allocated_gib"] for item in results])),
        "average_peak_memory_reserved_gib": float(np.mean([item["peak_memory_reserved_gib"] for item in results])),
        "average_total_flops": None,
        "average_total_flops_human": None,
        "average_flops_per_frame": None,
        "average_flops_per_frame_human": None,
    }

    flops_results = [item for item in results if "total_flops" in item]
    if flops_results:
        avg_total_flops = float(np.mean([item["total_flops"] for item in flops_results]))
        avg_flops_per_frame = float(np.mean([item["flops_per_frame"] for item in flops_results]))
        summary.update(
            {
                "average_total_flops": avg_total_flops,
                "average_total_flops_human": format_flops_human(avg_total_flops),
                "average_flops_per_frame": avg_flops_per_frame,
                "average_flops_per_frame_human": format_flops_human(avg_flops_per_frame),
            }
        )

    return summary


def build_profile_payload(args, total_clips, profile_results, skipped_clips):
    return {
        "profile_mode": args.profile_mode,
        "profile_scope": PROFILE_SCOPE,
        "json_path": args.json_path,
        "depth_method": args.depth_method,
        "resolution": {
            "width": int(args.resolution),
            "height": int(args.resolution // 2),
        },
        "profile_clip_limit": int(args.profile_clip_limit),
        "total_candidate_clips": int(total_clips),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "clips": profile_results,
        "summary": make_profile_summary(profile_results),
        "skipped_clips": skipped_clips,
        "notes": [
            "Metrics are collected per clip for the full VIPE pipeline: virtual camera setup, SLAM, panorama depth merge, and final depth stacking.",
            "GPU memory metrics are only meaningful when running on CUDA.",
            "FLOPs come from profiler-visible PyTorch ops only, so custom or non-PyTorch work may be undercounted.",
        ],
    }


def write_profile_payload(payload, output_path):
    if not output_path:
        return
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as file_obj:
        json.dump(payload, file_obj, indent=2)


def print_profile_result(result):
    lines = [
        f"[Profile:{result['clip_name']}] frames={result['num_frames']}",
        f"  wall_time_sec={result['wall_time_sec']:.3f}",
        (
            "  peak_memory_allocated="
            f"{result['peak_memory_allocated_gib']:.3f} GiB, "
            f"peak_memory_reserved={result['peak_memory_reserved_gib']:.3f} GiB"
        ),
    ]
    if "total_flops" in result:
        lines.append(
            f"  total_flops={result['total_flops_human']}, flops_per_frame={result['flops_per_frame_human']}"
        )
    print("\n".join(lines))


def run_vipe_inference(device, args):
    slam_cfg_default, virtual_cfg_default = load_vipe_configs()

    with open(args.json_path, "r") as file_obj:
        clips_raw = json.load(file_obj)

    all_clips = flatten_clips(clips_raw)
    print(f"Loaded {len(all_clips)} clips.")

    profile_results = []
    skipped_clips = []

    for clip_idx, clip in enumerate(all_clips, start=1):
        clip_name = clip["id"]
        print(f"\n=== Processing Clip {clip_idx}/{len(all_clips)}: {clip_name} ===")
        start_time = time.time()

        out_dir, npy_path, video_path = build_output_paths(args, clip_name)

        if not profile_enabled(args):
            os.makedirs(out_dir, exist_ok=True)
            if os.path.exists(npy_path):
                print("Result exists, skipping.")
                continue

        try:
            if profile_enabled(args):
                result = profile_clip(
                    device=device,
                    clip=clip,
                    slam_cfg_default=slam_cfg_default,
                    virtual_cfg_default=virtual_cfg_default,
                    args=args,
                )
                profile_results.append(result)
                print_profile_result(result)
                payload = build_profile_payload(args, len(all_clips), profile_results, skipped_clips)
                write_profile_payload(payload, args.profile_output_json)

                if len(profile_results) >= args.profile_clip_limit:
                    print(
                        f"Reached profile_clip_limit={args.profile_clip_limit}. "
                        "Stopping after required number of successful clips."
                    )
                    break
            else:
                with torch.inference_mode():
                    all_distances, original_rgbs = execute_vipe_clip(
                        device=device,
                        clip=clip,
                        slam_cfg_default=slam_cfg_default,
                        virtual_cfg_default=virtual_cfg_default,
                        args=args,
                        collect_visualization=True,
                    )
                save_clip_outputs(all_distances, original_rgbs, npy_path, video_path)
                elapsed_time = time.time() - start_time
                print(f" === Clip [{clip_name}] Done! Time: {elapsed_time:.2f}s ===")
                del all_distances, original_rgbs
                gc.collect()
        except Exception as exc:
            if profile_enabled(args):
                reason = str(exc) or exc.__class__.__name__
                skipped_entry = {
                    "clip_name": clip_name,
                    "town": clip["town"],
                    "path": clip["path"],
                    "reason": reason,
                }
                skipped_clips.append(skipped_entry)
                print(f"Skipping clip {clip_name}: {reason}")
                payload = build_profile_payload(args, len(all_clips), profile_results, skipped_clips)
                write_profile_payload(payload, args.profile_output_json)
                gc.collect()
                if device.type == "cuda":
                    # torch.cuda.empty_cache()
                    pass
                continue
            raise

    if profile_enabled(args):
        payload = build_profile_payload(args, len(all_clips), profile_results, skipped_clips)
        write_profile_payload(payload, args.profile_output_json)
        print("\n--- Profiling completed ---")
        print(json.dumps(payload["summary"], indent=2))
    else:
        print("\n--- All clips processed ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Path to the input JSON file defining clips")
    parser.add_argument("--image_base_dir", required=True, help="Base directory for images")
    parser.add_argument("--output_root_dir", default="vipe_results", help="Root directory for outputs")
    parser.add_argument("--resolution", type=int, default=1024, help="Width of the panorama (height will be width/2)")
    parser.add_argument(
        "--depth_method",
        type=str,
        default="depthcrafter",
        choices=["unik3d", "depthcrafter"],
        help="Depth estimation method to use",
    )
    parser.add_argument("--unet_path", type=str, default="tencent/DepthCrafter")
    parser.add_argument("--ppl_type", type=str, default="depthcrafter")
    parser.add_argument("--num_denoising_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--cpu_offload", type=str, default=None, choices=["model", "sequential", "None"])
    parser.add_argument(
        "--depthcrafter_npy_dir",
        type=str,
        default=None,
        help="Root directory for pre-computed DepthCrafter .npy results",
    )
    parser.add_argument(
        "--profile_mode",
        type=str,
        default=PROFILE_NONE,
        choices=[PROFILE_NONE, PROFILE_MEMORY, PROFILE_FLOPS, PROFILE_BOTH],
        help="Profiling mode: none, memory, flops, or both.",
    )
    parser.add_argument(
        "--profile_clip_limit",
        type=int,
        default=10,
        help="Number of successful clips to profile before stopping.",
    )
    parser.add_argument(
        "--profile_output_json",
        type=str,
        default=None,
        help="Optional JSON path to save profiling results.",
    )

    args = parser.parse_args()
    if args.cpu_offload == "None":
        args.cpu_offload = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing Vipe Inference...")
    run_vipe_inference(device, args)
