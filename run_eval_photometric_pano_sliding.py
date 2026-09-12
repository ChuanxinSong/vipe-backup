import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from tqdm import tqdm
try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    ssim_func = None

# Configuration
MASK_BOTTOM_RATIO = 0.15
DELTAS = [1, 2, 5, 10, 20]
SOURCE_STRIDE = 9

def panorama_to_xyz(depth, theta_range=(0, 2*np.pi), phi_range=(0, np.pi)):
    """
    depth: [B, 1, H, W] torch tensor
    Returns xyz: [B, 3, H, W]
    """
    B, _, H, W = depth.shape
    device = depth.device
    
    # torch.linspace doesn't have endpoint parameter, so we manually create the range
    # For theta: we want [0, 2*pi) with W points, so step = 2*pi/W
    theta = torch.arange(W, device=device, dtype=torch.float32) * (theta_range[1] - theta_range[0]) / W + theta_range[0]
    phi = torch.linspace(phi_range[0], phi_range[1], H, device=device)
    
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing='ij') # [H, W]
    
    x = depth * torch.sin(phi_grid) * torch.sin(theta_grid)
    y = -depth * torch.cos(phi_grid)
    z = depth * torch.sin(phi_grid) * torch.cos(theta_grid)
    
    return torch.cat([x, y, z], dim=1)

def xyz_to_panorama(xyz, theta_range=(0, 2*np.pi), phi_range=(0, np.pi)):
    """
    xyz: [B, 3, H, W]
    Returns uv: [B, H, W, 2] normalized to [-1, 1] for grid_sample
    """
    B, _, H, W = xyz.shape
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    
    r = torch.sqrt(x**2 + y**2 + z**2) + 1e-6
    
    phi = torch.acos(torch.clamp(-y/r, -1.0, 1.0))
    # atan2 returns values in [-pi, pi]. We want [0, 2pi].
    theta = torch.atan2(x, z)
    theta = torch.remainder(theta, 2 * np.pi)
    
    # Normalize to [-1, 1]
    u = 2.0 * (theta - theta_range[0]) / (theta_range[1] - theta_range[0]) - 1.0
    v = 2.0 * (phi - phi_range[0]) / (phi_range[1] - phi_range[0]) - 1.0
    
    return torch.stack([u, v], dim=-1)

def calculate_ssim(img1, img2, mask=None):
    """
    img1, img2: [H, W, 3] numpy arrays in [0, 255]
    mask: [H, W] binary numpy array
    """
    if ssim_func is None:
        return 0.0
    
    if mask is not None:
        # Compute full SSIM and mask it
        try:
             score, full_ssim = ssim_func(img1, img2, channel_axis=2, full=True, data_range=255)
             valid_pixels = full_ssim[mask > 0.5]
             return float(np.mean(valid_pixels) if len(valid_pixels) > 0 else 0.0)
        except Exception:
             score, full_ssim = ssim_func(img1, img2, multichannel=True, full=True, data_range=255)
             valid_pixels = full_ssim[mask > 0.5]
             return float(np.mean(valid_pixels) if len(valid_pixels) > 0 else 0.0)
    else:
        try:
            return float(ssim_func(img1, img2, channel_axis=2, data_range=255))
        except:
             return float(ssim_func(img1, img2, multichannel=True, data_range=255))

def calculate_psnr(img1, img2, mask=None):
    if mask is not None:
        valid_pixels = mask > 0.5
        if np.sum(valid_pixels) == 0:
            return 0.0
        mse = np.mean((img1[valid_pixels].astype(float) - img2[valid_pixels].astype(float)) ** 2)
    else:
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    if mse == 0:
        return 100.0
    
    return float(10 * np.log10(255.0**2 / mse))

def process_video(base_dir, video_name, output_dir, device):
    mask_bottom_ratio = MASK_BOTTOM_RATIO
    
    npy_path = os.path.join(base_dir, f"{video_name}_distance.npy")
    pose_path = os.path.join(base_dir, f"{video_name}_poses.json")
    video_path = os.path.join(base_dir, f"{video_name}_vis.mp4")
    
    if not os.path.exists(npy_path) or not os.path.exists(pose_path) or not os.path.exists(video_path):
        print(f"Skipping {video_name}, missing files.")
        return None

    # Create output directory for this clip
    clip_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(clip_output_dir, exist_ok=True)

    # Load data
    depths = np.load(npy_path) # [T, H, W]
    with open(pose_path, 'r') as f:
        poses = np.array(json.load(f)) # [T, 4, 4]
        
    vcap = cv2.VideoCapture(video_path)
    video_frames = []
    while True:
        ret, frame = vcap.read()
        if not ret: break
        h_total, w_total = frame.shape[:2]
        video_frames.append(frame[:h_total//2, :, ::-1].copy())
    vcap.release()
    
    # Align lengths
    T = min(len(depths), len(poses), len(video_frames))
    depths = depths[:T]
    poses = poses[:T]
    video_frames = video_frames[:T]
    
    H, W = depths.shape[1], depths.shape[2]
    
    video_metrics = []
    
    source_indices = list(range(0, T, SOURCE_STRIDE))
    
    # Prepare Mask
    frame_mask_torch = torch.ones(1, 1, H, W, device=device)
    bottom_start_row = int(H * (1 - mask_bottom_ratio))
    frame_mask_torch[:, :, bottom_start_row:, :] = 0
    
    for idx0 in source_indices:
        # Load source frame
        img0_torch = torch.from_numpy(video_frames[idx0]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        # Image masked by bottom part (ego car)
        img0_torch_masked = img0_torch * frame_mask_torch
        
        pose0 = torch.from_numpy(poses[idx0]).float().to(device)
        inv_pose0 = torch.inverse(pose0)
        
        for delta in DELTAS:
            idx1 = idx0 + delta
            if idx1 >= T: continue
            
            # Load Target frame
            img1_torch = torch.from_numpy(video_frames[idx1]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            depth1 = torch.from_numpy(depths[idx1]).unsqueeze(0).unsqueeze(0).to(device)
            pose1 = torch.from_numpy(poses[idx1]).float().to(device)
            
            # WARPING STRATEGY: 
            # We want to warp Source (0) to Target (1) view to compare with Target (1).
            # We unproject pixels of Target (1) using Depth1, transform to Source (0), and sample Source (0).
            
            xyz1 = panorama_to_xyz(depth1)
            
            # Transform P_target to P_source
            # P_source = T_source^-1 * T_target * P_target
            rel_pose = inv_pose0 @ pose1
            
            xyz1_flat = xyz1.view(1, 3, -1)
            xyz1_homo = torch.cat([xyz1_flat, torch.ones(1, 1, H*W, device=device)], dim=1)
            xyz0_from1_homo = torch.matmul(rel_pose, xyz1_homo)
            xyz0_from1 = xyz0_from1_homo[:, :3, :].view(1, 3, H, W)
            
            grid_backward = xyz_to_panorama(xyz0_from1)
            
            img0_warped = F.grid_sample(img0_torch_masked, grid_backward, mode='bilinear', 
                                         padding_mode='zeros', align_corners=True)
            
            # Also warp mask to check valid source regions
            mask_src_warped = F.grid_sample(frame_mask_torch, grid_backward, mode='nearest', 
                                           padding_mode='zeros', align_corners=True)
            
            radial_dist = torch.sqrt(xyz0_from1[:, 0]**2 + xyz0_from1[:, 1]**2 + xyz0_from1[:, 2]**2)
            valid_depth = radial_dist > 1e-6
            
            valid_grid = (grid_backward[..., 0] >= -1) & (grid_backward[..., 0] <= 1) & \
                         (grid_backward[..., 1] >= -1) & (grid_backward[..., 1] <= 1)
            
            valid_source = mask_src_warped.squeeze(1) > 0.5
            valid_target = frame_mask_torch.squeeze(1) > 0.5
            
            mask = valid_grid & valid_depth.squeeze(1) & valid_source & valid_target
            mask_np = mask.float().squeeze(0).cpu().numpy()
            
            img1_np = (img1_torch.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img0_warped_np = (img0_warped.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            ssim_val = calculate_ssim(img1_np, img0_warped_np, mask_np)
            psnr_val = calculate_psnr(img1_np, img0_warped_np, mask_np)
            
            l1_diff = np.abs(img1_np.astype(float) - img0_warped_np.astype(float)) / 255.0
            l1_val = np.float64(np.mean(l1_diff[mask_np > 0.5]) if np.sum(mask_np) > 0 else 1.0)
            
            coverage = np.float64(np.sum(mask_np > 0.5) / (H * W) * 100)
            
            step_metrics = {
                "step": int(delta), # Keeping 'step' for compatibility if needed, but 'delta' is better name
                "delta": int(delta),
                "source_idx": int(idx0),
                "target_idx": int(idx1),
                "ssim": ssim_val,
                "psnr": psnr_val,
                "masked_l1": l1_val,
                "coverage": coverage
            }
            video_metrics.append(step_metrics)
            
            # Save visualization
            vis_filename = f"{video_name}_src{idx0}_tgt{idx1}_delta{delta}.png"
            vis_path = os.path.join(clip_output_dir, vis_filename)
            
            img0_warped_with_holes = img0_warped_np.copy()
            img0_warped_with_holes[mask_np < 0.5] = [128, 128, 128]
            
            combined = np.vstack([img1_np, img0_warped_with_holes])
            cv2.imwrite(vis_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
    return video_metrics

def main():
    base_dir = "carla_benchmark_results/genex_results/genex_1024_unik3d"
    output_dir = os.path.join(base_dir, "evaluation_results_masked_sliding")
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    files = os.listdir(base_dir)
    video_names = sorted(list(set([f.replace("_distance.npy", "") for f in files if f.endswith("_distance.npy")])))
    
    all_results = {}
    
    print(f"Found {len(video_names)} videos to evaluate.")
    
    for video_name in tqdm(video_names):
        metrics = process_video(base_dir, video_name, output_dir, device)
        if metrics:
            all_results[video_name] = metrics
            
    # Save all results to JSON
    json_path = os.path.join(output_dir, "metrics.json")
    
    
    # Calculate averages by delta
    delta_metrics = {d: {'ssim': [], 'psnr': [], 'l1': [], 'coverage': []} for d in DELTAS}
    
    for vname, m_list in all_results.items():
        for m in m_list:
            d = m['delta']
            if d in delta_metrics:
                delta_metrics[d]['ssim'].append(m['ssim'])
                delta_metrics[d]['psnr'].append(m['psnr'])
                delta_metrics[d]['l1'].append(m['masked_l1'])
                delta_metrics[d]['coverage'].append(m['coverage'])
                
    summary = {}
    print("\n=== Evaluation Summary ===")
    
    overall_ssim = []
    overall_psnr = []
    overall_l1 = []
    overall_cov = []
    
    for d in DELTAS:
        dm = delta_metrics[d]
        if dm['ssim']:
            avg_ssim = float(np.mean(dm['ssim']))
            avg_psnr = float(np.mean(dm['psnr']))
            avg_l1 = float(np.mean(dm['l1']))
            avg_cov = float(np.mean(dm['coverage']))
            
            summary[f"delta_{d}"] = {
                "ssim": avg_ssim,
                "psnr": avg_psnr,
                "masked_l1": avg_l1,
                "coverage": avg_cov
            }
            print(f"Delta {d:2d}: SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.4f}, L1={avg_l1:.4f}, Cov={avg_cov:.2f}%")
            
            overall_ssim.extend(dm['ssim'])
            overall_psnr.extend(dm['psnr'])
            overall_l1.extend(dm['l1'])
            overall_cov.extend(dm['coverage'])
            
    if overall_ssim:
        avg_ssim_all = float(np.mean(overall_ssim))
        avg_psnr_all = float(np.mean(overall_psnr))
        avg_l1_all = float(np.mean(overall_l1))
        avg_cov_all = float(np.mean(overall_cov))
        
        summary["overall"] = {
            "ssim": avg_ssim_all,
            "psnr": avg_psnr_all,
            "masked_l1": avg_l1_all,
            "coverage": avg_cov_all
        }
        print(f"Overall : SSIM={avg_ssim_all:.4f}, PSNR={avg_psnr_all:.4f}, L1={avg_l1_all:.4f}, Cov={avg_cov_all:.2f}%")

    with open(json_path, 'w') as f:
        json.dump({"summary": summary, "video_results": all_results}, f, indent=4)
        
    print(f"Evaluation complete. Results saved to {json_path}")

if __name__ == "__main__":
    main()
