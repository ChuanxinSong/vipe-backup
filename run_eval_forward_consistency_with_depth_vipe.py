import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:  # pragma: no cover - best effort fallback
    ssim_func = None
    print("Warning: skimage not installed. SSIM will be reported as 0.0.")

# Configuration defaults (can be overridden via CLI)
MASK_BOTTOM_RATIO = 0.15
ANCHORS = [0, 20, 40, 60, 80]
# OFFSETS = [5, 10, 20, 50, 90]
# SAVE_OFFSETS = {20, 50, 90}

OFFSETS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
SAVE_OFFSETS = {5, 10, 20, 30, 40, 50, 60, 70, 80, 90}


def panorama_to_xyz(depth: torch.Tensor, theta_range=(-np.pi, np.pi), phi_range=(0, np.pi)) -> torch.Tensor:
    """Convert equirectangular depth map to 3D coordinates (B, 3, H, W)."""
    bsz, _, height, width = depth.shape
    device = depth.device

    theta = torch.arange(width, device=device, dtype=torch.float32)
    theta = theta * (theta_range[1] - theta_range[0]) / width + theta_range[0]
    phi = torch.linspace(phi_range[0], phi_range[1], height, device=device)

    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="ij")

    x = depth * torch.sin(phi_grid) * torch.sin(theta_grid)
    y = -depth * torch.cos(phi_grid)
    z = depth * torch.sin(phi_grid) * torch.cos(theta_grid)

    return torch.cat([x, y, z], dim=1)


def xyz_to_panorama(xyz: torch.Tensor, theta_range=(-np.pi, np.pi), phi_range=(0, np.pi)) -> torch.Tensor:
    """Project 3D coordinates to normalized UV grid (B, H, W, 2)."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-6

    phi = torch.acos(torch.clamp(-y / r, -1.0, 1.0))
    theta = torch.atan2(x, z)

    u = 2.0 * (theta - theta_range[0]) / (theta_range[1] - theta_range[0]) - 1.0
    v = 2.0 * (phi - phi_range[0]) / (phi_range[1] - phi_range[0]) - 1.0

    return torch.stack([u, v], dim=-1)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """SSIM on masked region; returns 0.0 if skimage is unavailable."""
    if ssim_func is None:
        return 0.0

    if mask is not None:
        score, full_ssim = ssim_func(img1, img2, channel_axis=2, full=True, data_range=255)
        valid_pixels = full_ssim[mask > 0.5]
        return float(np.mean(valid_pixels) if len(valid_pixels) > 0 else 0.0)

    return float(ssim_func(img1, img2, channel_axis=2, data_range=255))


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute PSNR on masked region."""
    if mask is not None:
        valid = mask > 0.5
        if np.sum(valid) == 0:
            return 0.0
        mse = np.mean((img1[valid].astype(float) - img2[valid].astype(float)) ** 2)
    else:
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)

    if mse == 0:
        return 100.0

    return float(10 * np.log10(255.0 ** 2 / mse))


def splat_colors_depth_zbuffer_gpu(
    u: torch.Tensor,
    v: torch.Tensor,
    depth: torch.Tensor,
    colors: torch.Tensor,
    height: int,
    width: int,
    atol: float = 1e-5,
    rtol: float = 1e-5,
):
    """Forward splat colors/depth via scatter_reduce_ with a Z-buffer."""
    device = u.device

    x_coords = torch.clamp(torch.round((u + 1.0) * 0.5 * (width - 1)), 0, width - 1).long()
    y_coords = torch.clamp(torch.round((v + 1.0) * 0.5 * (height - 1)), 0, height - 1).long()
    pixel_indices = y_coords * width + x_coords

    min_depth = torch.full((height * width,), float("inf"), device=device)
    min_depth.scatter_reduce_(0, pixel_indices, depth, reduce="amin", include_self=True)

    closest_depth = min_depth[pixel_indices]
    is_winner = torch.isclose(depth, closest_depth, atol=atol, rtol=rtol)

    synth_flat = torch.full((height * width, 3), 0.5, device=device)
    valid_mask_flat = torch.zeros(height * width, device=device, dtype=torch.bool)

    if is_winner.any():
        idx_winner = pixel_indices[is_winner]
        colors_winner = colors[is_winner]

        color_accum = torch.zeros((height * width, 3), device=device)
        count_accum = torch.zeros((height * width, 1), device=device)

        color_accum.scatter_add_(0, idx_winner.unsqueeze(1).expand(-1, 3), colors_winner)
        count_accum.scatter_add_(0, idx_winner.unsqueeze(1), torch.ones_like(colors_winner[:, :1]))

        valid_pixels = count_accum.squeeze(1) > 0
        synth_flat[valid_pixels] = color_accum[valid_pixels] / count_accum[valid_pixels]
        valid_mask_flat = valid_pixels

    return synth_flat, valid_mask_flat.to(torch.uint8), min_depth


def load_sequence_rgb(distance_root: Path, sequence_name: str):
    distance_path = distance_root / f"{sequence_name}_distance.npy"
    pose_path = distance_root / f"{sequence_name}_poses.json"
    video_path = distance_root / f"{sequence_name}_vis.mp4"

    if not distance_path.exists() or not pose_path.exists() or not video_path.exists():
        return None

    depths = np.load(distance_path)
    with open(pose_path, "r") as f:
        poses = np.array(json.load(f))

    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        height_total = frame.shape[0]
        frames.append(frame[: height_total // 2, :, ::-1].copy())
    capture.release()

    if len(frames) == 0:
        return None

    target_len = min(len(depths), len(poses), len(frames))
    return depths[:target_len], poses[:target_len], frames[:target_len]


def process_sequence(
    sequence_name: str,
    base_dir: Path,
    output_viz_dir: Path,
    device: torch.device,
    anchors: List[int],
    offsets: List[int],
    save_offsets: set,
) -> Optional[List[Dict[str, float]]]:
    loaded = load_sequence_rgb(base_dir, sequence_name)
    if loaded is None:
        print(f"Skipping {sequence_name}: missing or invalid data.")
        return None

    depths, poses, frames = loaded
    total_frames, height, width = depths.shape

    frame_mask = torch.ones(1, 1, height, width, device=device)
    bottom_start = int(height * (1 - MASK_BOTTOM_RATIO))
    frame_mask[:, :, bottom_start:, :] = 0
    frame_mask_np = frame_mask.squeeze(0).squeeze(0).cpu().numpy()
    frame_mask_flat = frame_mask.view(-1) > 0.5

    metrics: List[Dict[str, float]] = []

    for anchor_idx in anchors:
        if anchor_idx >= total_frames:
            continue

        img0 = torch.from_numpy(frames[anchor_idx].copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        depth0 = torch.from_numpy(depths[anchor_idx].copy()).unsqueeze(0).unsqueeze(0).to(device)
        pose0 = torch.from_numpy(poses[anchor_idx].copy()).float().to(device)

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

            img1_np = frames[target_idx]
            depth_target_np = depths[target_idx]
            pose1 = torch.from_numpy(poses[target_idx].copy()).float().to(device)

            synth_flat = torch.full((height * width, 3), 0.5, device=device)
            valid_mask_flat = torch.zeros(height * width, dtype=torch.uint8, device=device)
            depth_flat = torch.full((height * width,), float("inf"), device=device)

            with torch.no_grad():
                rel_pose = torch.inverse(pose1) @ pose0
                xyz1_from0_homo = torch.matmul(rel_pose, xyz0_homo)
                xyz1_from0 = xyz1_from0_homo[:3]

                radial = torch.sqrt(torch.sum(xyz1_from0 ** 2, dim=0))
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

                    synth_flat, valid_mask_flat, depth_flat = splat_colors_depth_zbuffer_gpu(
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

            depth_reproj_np = depth_flat.view(height, width).cpu().numpy()

            depth_valid_mask = (
                np.isfinite(depth_reproj_np)
                & (depth_reproj_np > 0.0)
                & np.isfinite(depth_target_np)
                & (depth_target_np > 0.0)
            )
            depth_mask = combined_mask & depth_valid_mask

            if np.sum(depth_mask) > 0:
                depth_target_vals = depth_target_np[depth_mask]
                depth_reproj_vals = depth_reproj_np[depth_mask]

                depth_absrel_val = float(
                    np.mean(np.abs(depth_target_vals - depth_reproj_vals) / np.maximum(depth_target_vals, 1e-6))
                )
                depth_ratio = np.maximum(
                    depth_target_vals / np.maximum(depth_reproj_vals, 1e-6),
                    depth_reproj_vals / np.maximum(depth_target_vals, 1e-6),
                )
                depth_delta_val = float(np.mean(depth_ratio < 1.25))
            else:
                depth_absrel_val = 0.0
                depth_delta_val = 0.0

            if offset in save_offsets:
                synth_visual = synth_img_uint8.copy()
                synth_visual[~combined_mask] = [128, 128, 128]

                error_map = np.mean(np.abs(img1_np.astype(float) - synth_img_uint8.astype(float)), axis=2)
                error_map = np.clip(error_map, 0.0, 255.0).astype(np.uint8)
                error_map_rgb = np.stack([error_map] * 3, axis=2)
                error_map_rgb[~combined_mask] = [128, 128, 128]

                depth_vis_mask = depth_mask if np.any(depth_mask) else depth_valid_mask
                if np.any(depth_vis_mask):
                    depth_min = float(np.min(depth_target_np[depth_vis_mask]))
                    depth_max = float(np.max(depth_target_np[depth_vis_mask]))
                    if depth_max - depth_min < 1e-6:
                        depth_max = depth_min + 1e-6
                else:
                    depth_min = 0.0
                    depth_max = 1.0

                depth_range = depth_max - depth_min
                depth_target_safe = np.where(np.isfinite(depth_target_np), depth_target_np, depth_min)
                depth_reproj_safe = np.where(np.isfinite(depth_reproj_np), depth_reproj_np, depth_max)

                depth_target_norm = np.clip((depth_target_safe - depth_min) / depth_range, 0.0, 1.0)
                depth_reproj_norm = np.clip((depth_reproj_safe - depth_min) / depth_range, 0.0, 1.0)

                depth_target_rgb = (depth_target_norm[..., None] * 255).astype(np.uint8).repeat(3, axis=2)
                depth_reproj_rgb = (depth_reproj_norm[..., None] * 255).astype(np.uint8).repeat(3, axis=2)

                depth_absrel_map = np.zeros_like(depth_target_np, dtype=np.float32)
                if np.any(depth_mask):
                    depth_absrel_map[depth_mask] = (
                        np.abs(depth_target_np[depth_mask] - depth_reproj_np[depth_mask])
                        / np.maximum(depth_target_np[depth_mask], 1e-6)
                    )
                depth_absrel_vis = np.clip(depth_absrel_map, 0.0, 1.0)
                depth_absrel_rgb = (depth_absrel_vis[..., None] * 255).astype(np.uint8).repeat(3, axis=2)

                invalid_depth = ~depth_mask
                depth_target_rgb[invalid_depth] = [128, 128, 128]
                depth_reproj_rgb[invalid_depth] = [128, 128, 128]

                # Depth AbsRel map keeps the same invalid mask for consistency.
                depth_absrel_rgb[invalid_depth] = [128, 128, 128]

                row_rgb = np.hstack([img1_np, synth_visual, error_map_rgb])
                row_depth = np.hstack([depth_target_rgb, depth_reproj_rgb, depth_absrel_rgb])
                combined = np.vstack([row_rgb, row_depth])

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
                    "depth_absrel": depth_absrel_val,
                    "depth_delta_1_25": depth_delta_val,
                }
            )

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate forward consistency with ViPE distance/pose outputs using forward splatting."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="carla_benchmark_results/genex_results/genex_1024_unik3d",
        help="Directory containing *_distance.npy, *_poses.json, *_vis.mp4 for each sequence.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Directory to store metrics and visualizations (defaults to base_dir/forward_consistency_results).",
    )
    parser.add_argument(
        "--anchors",
        type=int,
        nargs="+",
        default=ANCHORS,
        help="Anchor frame indices (default matches DepthCrafter template).",
    )
    parser.add_argument(
        "--offsets",
        type=int,
        nargs="+",
        default=OFFSETS,
        help="Temporal offsets evaluated from each anchor (default matches DepthCrafter template).",
    )
    parser.add_argument(
        "--save-offsets",
        type=int,
        nargs="+",
        default=None,
        help="Offsets that trigger visualization exports (defaults to template SAVE_OFFSETS).",
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
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory {base_dir} does not exist.")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else base_dir / "forward_consistency_results_1"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_viz_root = output_root / "visualizations"
    output_viz_root.mkdir(parents=True, exist_ok=True)

    anchors = sorted(set(args.anchors))
    offsets = sorted(set(args.offsets))
    save_offsets = set(args.save_offsets) if args.save_offsets is not None else SAVE_OFFSETS

    device = torch.device(args.device)
    print(f"Using device: {device}")

    sequence_files = sorted(base_dir.glob("*_distance.npy"))
    sequence_names = [f.stem.replace("_distance", "") for f in sequence_files]

    print(f"Found {len(sequence_names)} sequences in {base_dir}.")

    all_metrics: Dict[str, List[Dict[str, float]]] = {}

    for sequence_name in tqdm(sequence_names, desc="Evaluating sequences"):
        try:
            seq_metrics = process_sequence(
                sequence_name,
                base_dir,
                output_viz_root,
                device,
                anchors,
                offsets,
                save_offsets,
            )
            if seq_metrics:
                all_metrics[sequence_name] = seq_metrics
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Error processing {sequence_name}: {exc}")
            import traceback

            traceback.print_exc()

    metrics_path = output_root / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Evaluation complete. Results saved to {metrics_path}")

    all_ssim = []
    all_psnr = []
    all_l1 = []
    all_cov = []
    all_depth_absrel = []
    all_depth_delta = []

    for clip_metrics in all_metrics.values():
        for metric in clip_metrics:
            all_ssim.append(metric["ssim"])
            all_psnr.append(metric["psnr"])
            all_l1.append(metric["masked_l1"])
            all_cov.append(metric["coverage"])
            all_depth_absrel.append(metric["depth_absrel"])
            all_depth_delta.append(metric["depth_delta_1_25"])

    if all_ssim:
        print(f"Overall Average SSIM: {np.mean(all_ssim):.4f}")
        print(f"Overall Average PSNR: {np.mean(all_psnr):.4f}")
        print(f"Overall Average L1: {np.mean(all_l1):.4f}")
        print(f"Overall Average Coverage: {np.mean(all_cov):.4f}%")
        print(f"Overall Average Depth AbsRel: {np.mean(all_depth_absrel):.4f}")
        print(f"Overall Average Depth delta<1.25: {np.mean(all_depth_delta):.4f}")


if __name__ == "__main__":
    main()
