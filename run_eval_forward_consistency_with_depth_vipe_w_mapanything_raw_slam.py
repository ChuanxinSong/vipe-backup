import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from run_eval_forward_consistency_with_depth_vipe_w_raw_slam import (
    ALIGN_MAX_DISTANCE,
    ALIGN_MIN_DISTANCE,
    ANCHORS,
    MASK_BOTTOM_RATIO,
    OFFSETS,
    SAVE_OFFSETS,
    calculate_psnr,
    calculate_ssim,
    load_vipe_depth_sequence,
    panorama_to_xyz,
    read_rgb_frames,
    resize_depth_sequence,
    splat_colors_depth_zbuffer_gpu,
    xyz_to_panorama,
)


MAPANYTHING_RAW_SLAM_DIR_DEFAULT = (
    "/home/user/songcx/code/DepthCrafter/"
    "map-anything/carla_benchmark_results/mapanything_results_raw_slam/mapanything_1024_raw_slam"
)


def load_mapanything_sequence(mapanything_dir: Path, sequence_name: str):
    distance_path = mapanything_dir / f"{sequence_name}_distance.npy"
    valid_mask_path = mapanything_dir / f"{sequence_name}_distance_valid_mask.npy"
    pose_path = mapanything_dir / f"{sequence_name}_poses.json"
    video_path = mapanything_dir / f"{sequence_name}_vis.mp4"

    missing = [str(path) for path in [distance_path, valid_mask_path, pose_path, video_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MapAnything raw_slam files for {sequence_name}: {missing}")

    depths = np.load(distance_path).astype(np.float32, copy=False)
    valid_masks = np.load(valid_mask_path).astype(np.uint8, copy=False)
    with pose_path.open("r") as f:
        poses = np.array(json.load(f), dtype=np.float32)
    frames = read_rgb_frames(video_path)

    if depths.ndim != 3:
        raise ValueError(f"{distance_path} must have shape [T, H, W], got {depths.shape}")
    if valid_masks.shape != depths.shape:
        raise ValueError(
            f"{valid_mask_path} must have the same shape as MapAnything depths. "
            f"Got {valid_masks.shape} vs {depths.shape}."
        )
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{pose_path} must have shape [T, 4, 4], got {poses.shape}")

    frame_shape = frames[0].shape[:2]
    if frame_shape != depths.shape[1:]:
        raise ValueError(
            f"MapAnything video frame size for {sequence_name} does not match depth size: "
            f"{frame_shape} vs {depths.shape[1:]}"
        )

    lengths = {
        "mapanything_depth": len(depths),
        "mapanything_valid_mask": len(valid_masks),
        "mapanything_pose": len(poses),
        "mapanything_video": len(frames),
    }
    expected_len = lengths["mapanything_depth"]
    mismatched = {key: value for key, value in lengths.items() if value != expected_len}
    if mismatched:
        raise ValueError(f"Length mismatch in MapAnything data for {sequence_name}: {lengths}")

    return depths, valid_masks, poses, frames


def align_vipe_depth_to_mapanything(
    sequence_name: str,
    vipe_depths: np.ndarray,
    map_depths: np.ndarray,
    map_valid_masks: np.ndarray,
) -> Tuple[np.ndarray, Optional[float], Optional[float], bool]:
    if vipe_depths.ndim != 3:
        raise ValueError(f"ViPE depths for {sequence_name} must have shape [T, H, W], got {vipe_depths.shape}")

    if len(vipe_depths) != len(map_depths):
        raise ValueError(
            f"Length mismatch for {sequence_name}: "
            f"ViPE depth has {len(vipe_depths)} frames, MapAnything depth has {len(map_depths)} frames."
        )

    vipe_resized = resize_depth_sequence(vipe_depths, map_depths.shape[1:])
    clipped_original = np.clip(vipe_resized, 0.0, 1000.0).astype(np.float32, copy=False)

    x_list = []
    y_list = []

    for frame_idx in range(len(map_depths)):
        vipe_depth = vipe_resized[frame_idx]
        map_depth = map_depths[frame_idx]
        map_valid = map_valid_masks[frame_idx] > 0

        mask = (
            map_valid
            & np.isfinite(map_depth)
            & (map_depth > ALIGN_MIN_DISTANCE)
            & (map_depth < ALIGN_MAX_DISTANCE)
            & np.isfinite(vipe_depth)
            & (vipe_depth > ALIGN_MIN_DISTANCE)
            & (vipe_depth < ALIGN_MAX_DISTANCE)
        )

        if np.any(mask):
            x_list.append(vipe_depth[mask])
            y_list.append(map_depth[mask])

    if not x_list:
        print(f"Warning: No valid MapAnything alignment points found for {sequence_name}, keeping original depths.")
        return clipped_original, None, None, False

    x_all = np.concatenate(x_list).astype(np.float64, copy=False)
    y_all = np.concatenate(y_list).astype(np.float64, copy=False)
    design = np.vstack([x_all, np.ones(len(x_all), dtype=np.float64)]).T
    try:
        scale, shift = np.linalg.lstsq(design, y_all, rcond=None)[0]
    except np.linalg.LinAlgError:
        print(f"Warning: MapAnything alignment failed for {sequence_name}, keeping original depths.")
        return clipped_original, None, None, False

    if not np.isfinite(scale) or not np.isfinite(shift):
        print(
            f"Warning: MapAnything alignment produced non-finite parameters for {sequence_name}, "
            "keeping original depths."
        )
        return clipped_original, None, None, False

    aligned_depths = vipe_resized.astype(np.float32, copy=True)
    aligned_depths = aligned_depths * float(scale) + float(shift)
    aligned_depths = np.clip(aligned_depths, 0.0, 1000.0).astype(np.float32, copy=False)

    return aligned_depths, float(scale), float(shift), True


def process_sequence(
    sequence_name: str,
    vipe_dir: Path,
    mapanything_dir: Path,
    output_viz_dir: Path,
    device: torch.device,
    anchors: List[int],
    offsets: List[int],
    save_offsets: set,
) -> List[Dict[str, float]]:
    vipe_depths = load_vipe_depth_sequence(vipe_dir, sequence_name)
    map_depths, map_valid_masks, map_poses, map_frames = load_mapanything_sequence(mapanything_dir, sequence_name)

    if len(vipe_depths) != len(map_depths):
        raise ValueError(
            f"Length mismatch for {sequence_name}: "
            f"ViPE depth has {len(vipe_depths)} frames, MapAnything depth has {len(map_depths)} frames."
        )

    aligned_depths, align_scale, align_shift, did_align = align_vipe_depth_to_mapanything(
        sequence_name,
        vipe_depths,
        map_depths,
        map_valid_masks,
    )
    if did_align:
        print(
            f"{sequence_name}: aligned ViPE depth to MapAnything with "
            f"scale={align_scale:.6f}, shift={align_shift:.6f}"
        )
    else:
        print(f"{sequence_name}: MapAnything alignment failed, using clipped ViPE depth without alignment.")

    total_frames, height, width = aligned_depths.shape

    frame_mask = torch.ones(1, 1, height, width, device=device)
    bottom_start = int(height * (1 - MASK_BOTTOM_RATIO))
    frame_mask[:, :, bottom_start:, :] = 0
    frame_mask_np = frame_mask.squeeze(0).squeeze(0).cpu().numpy()
    frame_mask_flat = frame_mask.view(-1) > 0.5

    metrics: List[Dict[str, float]] = []

    for anchor_idx in anchors:
        if anchor_idx >= total_frames:
            continue

        img0 = (
            torch.from_numpy(map_frames[anchor_idx].copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        )
        depth0 = torch.from_numpy(aligned_depths[anchor_idx].copy()).unsqueeze(0).unsqueeze(0).to(device)
        pose0 = torch.from_numpy(map_poses[anchor_idx].copy()).float().to(device)

        img0_masked = img0 * frame_mask

        xyz0 = panorama_to_xyz(depth0)
        xyz0_flat = xyz0.view(3, -1)
        ones = torch.ones(1, xyz0_flat.shape[1], device=device)
        xyz0_homo = torch.cat([xyz0_flat, ones], dim=0)

        colors_flat = img0_masked.squeeze(0).view(3, -1)
        source_depth_flat = depth0.view(-1)
        source_mask_flat = frame_mask_flat & torch.isfinite(source_depth_flat) & (source_depth_flat > 0)

        for offset in offsets:
            target_idx = anchor_idx + offset
            if target_idx >= total_frames:
                continue

            img1_np = map_frames[target_idx]
            pose1 = torch.from_numpy(map_poses[target_idx].copy()).float().to(device)

            synth_flat = torch.full((height * width, 3), 0.5, device=device)
            valid_mask_flat = torch.zeros(height * width, dtype=torch.uint8, device=device)

            with torch.no_grad():
                rel_pose = torch.inverse(pose1) @ pose0
                xyz1_from0_homo = torch.matmul(rel_pose, xyz0_homo)
                xyz1_from0 = xyz1_from0_homo[:3]

                radial = torch.sqrt(torch.sum(xyz1_from0**2, dim=0))
                xyz1_from0_reshaped = xyz1_from0.view(1, 3, height, width)
                grid_forward = xyz_to_panorama(xyz1_from0_reshaped)

                u = grid_forward[0, :, :, 0].reshape(-1)
                v = grid_forward[0, :, :, 1].reshape(-1)

                valid_coords = (u >= -1.0) & (u <= 1.0) & (v >= -1.0) & (v <= 1.0)
                valid_depth = torch.isfinite(radial) & (radial > 1e-6)
                valid_uv = torch.isfinite(u) & torch.isfinite(v)
                valid_points = source_mask_flat & valid_coords & valid_depth & valid_uv

                if valid_points.any():
                    u_valid = u[valid_points]
                    v_valid = v[valid_points]
                    radial_valid = radial[valid_points]
                    colors_valid = colors_flat[:, valid_points].T.contiguous()

                    synth_flat, valid_mask_flat, _ = splat_colors_depth_zbuffer_gpu(
                        u_valid, v_valid, radial_valid, colors_valid, height, width
                    )

            synth_img_np = torch.clamp(synth_flat.view(height, width, 3), 0.0, 1.0).cpu().numpy()
            synth_img_uint8 = (synth_img_np * 255).astype(np.uint8)

            valid_mask_np = valid_mask_flat.view(height, width).cpu().numpy()
            combined_mask = (valid_mask_np > 0) & (frame_mask_np > 0.5)
            mask_float = combined_mask.astype(np.float32)

            ssim_val = calculate_ssim(img1_np, synth_img_uint8, mask_float)
            psnr_val = calculate_psnr(img1_np, synth_img_uint8, mask_float)

            l1_diff = np.abs(img1_np.astype(float) - synth_img_uint8.astype(float)) / 255.0
            l1_val = float(np.mean(l1_diff[combined_mask]) if np.sum(combined_mask) > 0 else 1.0)
            coverage = float(np.mean(combined_mask) * 100.0)

            if offset in save_offsets:
                synth_visual = synth_img_uint8.copy()
                synth_visual[~combined_mask] = [128, 128, 128]

                error_map = np.mean(np.abs(img1_np.astype(float) - synth_img_uint8.astype(float)), axis=2)
                error_map = np.clip(error_map, 0.0, 255.0).astype(np.uint8)
                error_map_rgb = np.stack([error_map] * 3, axis=2)
                error_map_rgb[~combined_mask] = [128, 128, 128]

                combined = np.hstack([img1_np, synth_visual, error_map_rgb])

                viz_dir = output_viz_dir / sequence_name
                viz_dir.mkdir(parents=True, exist_ok=True)

                vis_filename = f"{sequence_name}_src{anchor_idx}_t{target_idx}_off{offset}.png"
                cv2.imwrite(str(viz_dir / vis_filename), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            metrics.append(
                {
                    "source_frame_idx": int(anchor_idx),
                    "offset": int(offset),
                    "step": int(offset),
                    "target_frame_idx": int(target_idx),
                    "ssim": ssim_val,
                    "psnr": psnr_val,
                    "masked_l1": l1_val,
                    "coverage": coverage,
                }
            )

    return metrics


def collect_sequence_names(vipe_dir: Path, mapanything_dir: Path) -> List[str]:
    vipe_sequences = {path.stem.replace("_distance", "") for path in vipe_dir.glob("*_distance.npy")}
    mapanything_sequences = {path.stem.replace("_distance", "") for path in mapanything_dir.glob("*_distance.npy")}

    common_sequences = sorted(vipe_sequences & mapanything_sequences)
    if not common_sequences:
        raise ValueError(f"No common sequences found between {vipe_dir} and {mapanything_dir}.")

    common_sequence_set = set(common_sequences)
    only_vipe = sorted(vipe_sequences - common_sequence_set)
    only_mapanything = sorted(mapanything_sequences - common_sequence_set)
    if only_vipe:
        print(f"Skipping {len(only_vipe)} ViPE-only sequences: {only_vipe[:10]}")
    if only_mapanything:
        print(f"Skipping {len(only_mapanything)} MapAnything-only sequences: {only_mapanything[:10]}")

    return common_sequences


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate forward consistency using ViPE depth aligned to MapAnything raw_slam distance and poses."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="carla_benchmark_results/genex_results/genex_1024_unik3d",
        help="Directory containing ViPE *_distance.npy files.",
    )
    parser.add_argument(
        "--mapanything_dir",
        type=str,
        default=MAPANYTHING_RAW_SLAM_DIR_DEFAULT,
        help="Directory containing MapAnything raw_slam *_distance.npy, *_distance_valid_mask.npy, *_poses.json, *_vis.mp4.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Directory to store metrics and visualizations.",
    )
    parser.add_argument(
        "--anchors",
        type=int,
        nargs="+",
        default=ANCHORS,
        help="Anchor frame indices.",
    )
    parser.add_argument(
        "--offsets",
        type=int,
        nargs="+",
        default=OFFSETS,
        help="Temporal offsets evaluated from each anchor.",
    )
    parser.add_argument(
        "--save-offsets",
        type=int,
        nargs="+",
        default=None,
        help="Offsets that trigger visualization exports.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device identifier.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    mapanything_dir = Path(args.mapanything_dir).expanduser().resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"ViPE base directory {base_dir} does not exist.")
    if not mapanything_dir.exists():
        raise FileNotFoundError(f"MapAnything directory {mapanything_dir} does not exist.")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else base_dir / "forward_consistency_results_mapanything_raw_slam_align"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_viz_root = output_root / "visualizations"
    output_viz_root.mkdir(parents=True, exist_ok=True)

    anchors = sorted(set(args.anchors))
    offsets = sorted(set(args.offsets))
    save_offsets = set(args.save_offsets) if args.save_offsets is not None else SAVE_OFFSETS

    device = torch.device(args.device)
    print(f"Using device: {device}")

    sequence_names = collect_sequence_names(base_dir, mapanything_dir)
    print(f"Found {len(sequence_names)} common sequences.")

    all_metrics: Dict[str, List[Dict[str, float]]] = {}

    for sequence_name in tqdm(sequence_names, desc="Evaluating sequences"):
        seq_metrics = process_sequence(
            sequence_name,
            base_dir,
            mapanything_dir,
            output_viz_root,
            device,
            anchors,
            offsets,
            save_offsets,
        )
        all_metrics[sequence_name] = seq_metrics

    metrics_path = output_root / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Evaluation complete. Results saved to {metrics_path}")

    all_ssim = []
    all_psnr = []
    all_l1 = []
    all_cov = []

    for clip_metrics in all_metrics.values():
        for metric in clip_metrics:
            all_ssim.append(metric["ssim"])
            all_psnr.append(metric["psnr"])
            all_l1.append(metric["masked_l1"])
            all_cov.append(metric["coverage"])

    if all_ssim:
        print(f"Overall Average SSIM: {np.mean(all_ssim):.4f}")
        print(f"Overall Average PSNR: {np.mean(all_psnr):.4f}")
        print(f"Overall Average L1: {np.mean(all_l1):.4f}")
        print(f"Overall Average Coverage: {np.mean(all_cov):.4f}%")


if __name__ == "__main__":
    main()
