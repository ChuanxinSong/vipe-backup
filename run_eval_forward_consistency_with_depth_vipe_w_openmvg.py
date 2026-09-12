import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
OFFSETS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
SAVE_OFFSETS = {5, 10, 20, 30, 40, 50, 60, 70, 80, 90}
EPSILON = 1e-6
ALIGN_MIN_DISTANCE = 1e-1
ALIGN_MAX_DISTANCE = 20.0


def panorama_to_xyz(depth: torch.Tensor, theta_range=(-np.pi, np.pi), phi_range=(0, np.pi)) -> torch.Tensor:
    """Convert equirectangular depth map to 3D coordinates (B, 3, H, W)."""
    _, _, height, width = depth.shape
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

    r = torch.sqrt(x**2 + y**2 + z**2) + 1e-6

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

    return float(10 * np.log10(255.0**2 / mse))


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


def read_rgb_frames(video_path: Path) -> List[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        height_total = frame.shape[0]
        frames.append(frame[: height_total // 2, :, ::-1].copy())

    capture.release()

    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")

    return frames


def resize_depth_sequence(depths: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if depths.shape[1:] == (target_h, target_w):
        return depths.astype(np.float32, copy=False)

    resized = np.empty((depths.shape[0], target_h, target_w), dtype=np.float32)
    for idx, depth in enumerate(depths):
        resized[idx] = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized


def load_vipe_depth_sequence(vipe_dir: Path, sequence_name: str) -> np.ndarray:
    distance_path = vipe_dir / f"{sequence_name}_distance.npy"
    if not distance_path.exists():
        raise FileNotFoundError(f"Missing ViPE distance file: {distance_path}")
    return np.load(distance_path).astype(np.float32, copy=False)


def load_openmvg_sequence(openmvg_dir: Path, sequence_name: str):
    distance_path = openmvg_dir / f"{sequence_name}_distance.npy"
    valid_mask_path = openmvg_dir / f"{sequence_name}_distance_valid_mask.npy"
    pose_path = openmvg_dir / f"{sequence_name}_poses.json"
    video_path = openmvg_dir / f"{sequence_name}_vis.mp4"

    missing = [str(path) for path in [distance_path, valid_mask_path, pose_path, video_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing openMVG files for {sequence_name}: {missing}")

    depths = np.load(distance_path).astype(np.float32, copy=False)
    valid_masks = np.load(valid_mask_path).astype(np.uint8, copy=False)
    with pose_path.open("r") as f:
        poses = np.array(json.load(f), dtype=np.float32)
    frames = read_rgb_frames(video_path)

    if depths.ndim != 3:
        raise ValueError(f"{distance_path} must have shape [T, H, W], got {depths.shape}")
    if valid_masks.shape != depths.shape:
        raise ValueError(
            f"{valid_mask_path} must have the same shape as openMVG depths. "
            f"Got {valid_masks.shape} vs {depths.shape}."
        )
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{pose_path} must have shape [T, 4, 4], got {poses.shape}")

    frame_shape = frames[0].shape[:2]
    if frame_shape != depths.shape[1:]:
        raise ValueError(
            f"openMVG video frame size for {sequence_name} does not match openMVG depth size: "
            f"{frame_shape} vs {depths.shape[1:]}"
        )

    lengths = {
        "openmvg_depth": len(depths),
        "openmvg_valid_mask": len(valid_masks),
        "openmvg_pose": len(poses),
        "openmvg_video": len(frames),
    }
    expected_len = lengths["openmvg_depth"]
    mismatched = {key: value for key, value in lengths.items() if value != expected_len}
    if mismatched:
        raise ValueError(f"Length mismatch in openMVG data for {sequence_name}: {lengths}")

    return depths, valid_masks, poses, frames


def align_vipe_depth_to_openmvg(
    sequence_name: str,
    vipe_depths: np.ndarray,
    openmvg_depths: np.ndarray,
    openmvg_valid_masks: np.ndarray,
) -> Tuple[np.ndarray, Optional[float], Optional[float], bool]:
    if vipe_depths.ndim != 3:
        raise ValueError(f"ViPE depths for {sequence_name} must have shape [T, H, W], got {vipe_depths.shape}")

    if len(vipe_depths) != len(openmvg_depths):
        raise ValueError(
            f"Length mismatch for {sequence_name}: "
            f"ViPE depth has {len(vipe_depths)} frames, openMVG depth has {len(openmvg_depths)} frames."
        )

    vipe_resized = resize_depth_sequence(vipe_depths, openmvg_depths.shape[1:])
    clipped_original = np.clip(vipe_resized, 0.0, 1000.0).astype(np.float32, copy=False)

    x_list = []
    y_list = []

    for frame_idx in range(len(openmvg_depths)):
        vipe_depth = vipe_resized[frame_idx]
        openmvg_depth = openmvg_depths[frame_idx]
        openmvg_valid = openmvg_valid_masks[frame_idx] > 0

        mask = (
            openmvg_valid
            & np.isfinite(openmvg_depth)
            & (openmvg_depth > ALIGN_MIN_DISTANCE)
            & (openmvg_depth < ALIGN_MAX_DISTANCE)
            & np.isfinite(vipe_depth)
            & (vipe_depth > ALIGN_MIN_DISTANCE)
            & (vipe_depth < ALIGN_MAX_DISTANCE)
        )

        if np.any(mask):
            x_list.append(vipe_depth[mask])
            y_list.append(openmvg_depth[mask])

    if not x_list:
        print(f"Warning: No valid openMVG alignment points found for {sequence_name}, keeping original depths.")
        return clipped_original, None, None, False

    x_all = np.concatenate(x_list).astype(np.float64, copy=False)
    y_all = np.concatenate(y_list).astype(np.float64, copy=False)
    design = np.vstack([x_all, np.ones(len(x_all), dtype=np.float64)]).T
    try:
        scale, shift = np.linalg.lstsq(design, y_all, rcond=None)[0]
    except np.linalg.LinAlgError:
        print(f"Warning: openMVG alignment failed for {sequence_name}, keeping original depths.")
        return clipped_original, None, None, False

    if not np.isfinite(scale) or not np.isfinite(shift):
        print(
            f"Warning: openMVG alignment produced non-finite parameters for {sequence_name}, "
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
    openmvg_dir: Path,
    output_viz_dir: Path,
    device: torch.device,
    anchors: List[int],
    offsets: List[int],
    save_offsets: set,
) -> List[Dict[str, float]]:
    vipe_depths = load_vipe_depth_sequence(vipe_dir, sequence_name)
    openmvg_depths, openmvg_valid_masks, openmvg_poses, openmvg_frames = load_openmvg_sequence(
        openmvg_dir,
        sequence_name,
    )

    if len(vipe_depths) != len(openmvg_depths):
        raise ValueError(
            f"Length mismatch for {sequence_name}: "
            f"ViPE depth has {len(vipe_depths)} frames, openMVG depth has {len(openmvg_depths)} frames."
        )

    aligned_depths, align_scale, align_shift, did_align = align_vipe_depth_to_openmvg(
        sequence_name,
        vipe_depths,
        openmvg_depths,
        openmvg_valid_masks,
    )
    if did_align:
        print(
            f"{sequence_name}: aligned ViPE depth to openMVG with scale={align_scale:.6f}, shift={align_shift:.6f}"
        )
    else:
        print(f"{sequence_name}: openMVG alignment failed, using clipped ViPE depth without alignment.")

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
            torch.from_numpy(openmvg_frames[anchor_idx].copy()).permute(2, 0, 1).float().unsqueeze(0).to(device)
            / 255.0
        )
        depth0 = torch.from_numpy(aligned_depths[anchor_idx].copy()).unsqueeze(0).unsqueeze(0).to(device)
        pose0 = torch.from_numpy(openmvg_poses[anchor_idx].copy()).float().to(device)

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

            img1_np = openmvg_frames[target_idx]
            pose1 = torch.from_numpy(openmvg_poses[target_idx].copy()).float().to(device)

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


def collect_sequence_names(vipe_dir: Path, openmvg_dir: Path) -> List[str]:
    vipe_sequences = {path.stem.replace("_distance", "") for path in vipe_dir.glob("*_distance.npy")}
    openmvg_sequences = {path.stem.replace("_distance", "") for path in openmvg_dir.glob("*_distance.npy")}

    common_sequences = sorted(vipe_sequences & openmvg_sequences)
    if not common_sequences:
        raise ValueError(f"No common sequences found between {vipe_dir} and {openmvg_dir}.")

    only_vipe = sorted(vipe_sequences - openmvg_sequences)
    only_openmvg = sorted(openmvg_sequences - vipe_sequences)
    if only_vipe:
        print(f"Skipping {len(only_vipe)} ViPE-only sequences: {only_vipe[:10]}")
    if only_openmvg:
        print(f"Skipping {len(only_openmvg)} openMVG-only sequences: {only_openmvg[:10]}")

    return common_sequences


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate forward consistency using ViPE depth aligned to openMVG sparse distance and openMVG poses."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="carla_benchmark_results/genex_results/genex_1024_unik3d",
        help="Directory containing ViPE *_distance.npy files.",
    )
    parser.add_argument(
        "--openmvg_dir",
        type=str,
        default="openMVG/carla_benchmark_results/genex_results_raw_slam_openmvg/genex_1024_raw_slam_openmvg",
        help="Directory containing openMVG *_distance.npy, *_distance_valid_mask.npy, *_poses.json, *_vis.mp4.",
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
    openmvg_dir = Path(args.openmvg_dir).expanduser().resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"ViPE base directory {base_dir} does not exist.")
    if not openmvg_dir.exists():
        raise FileNotFoundError(f"openMVG directory {openmvg_dir} does not exist.")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else base_dir / "forward_consistency_results_openmvg_align"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_viz_root = output_root / "visualizations"
    output_viz_root.mkdir(parents=True, exist_ok=True)

    anchors = sorted(set(args.anchors))
    offsets = sorted(set(args.offsets))
    save_offsets = set(args.save_offsets) if args.save_offsets is not None else SAVE_OFFSETS

    device = torch.device(args.device)
    print(f"Using device: {device}")

    sequence_names = collect_sequence_names(base_dir, openmvg_dir)
    print(f"Found {len(sequence_names)} common sequences.")

    all_metrics: Dict[str, List[Dict[str, float]]] = {}

    for sequence_name in tqdm(sequence_names, desc="Evaluating sequences"):
        seq_metrics = process_sequence(
            sequence_name,
            base_dir,
            openmvg_dir,
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
