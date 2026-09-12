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
from torchmetrics.image import StructuralSimilarityIndexMeasure

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
    
    # Vipe Equirectangular projection:
    # x = sin(phi) * sin(theta)
    # y = -cos(phi)
    # z = sin(phi) * cos(theta)
    
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
    
    # phi = acos(-y/r)
    phi = torch.acos(torch.clamp(-y/r, -1.0, 1.0))
    # theta = atan2(x, z)
    theta = torch.atan2(x, z)
    # atan2 returns values in [-pi, pi]. We want [0, 2pi].
    theta = torch.remainder(theta, 2 * np.pi)
    
    # Normalize to [-1, 1]
    # theta: [0, 2pi] -> [-1, 1]
    u = 2.0 * (theta - theta_range[0]) / (theta_range[1] - theta_range[0]) - 1.0
    # phi: [0, pi] -> [-1, 1]
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
        score, full_ssim = ssim_func(img1, img2, channel_axis=2, full=True, data_range=255)
        valid_pixels = full_ssim[mask > 0.5]
        return np.mean(valid_pixels) if len(valid_pixels) > 0 else 0.0
    else:
        return ssim_func(img1, img2, channel_axis=2, data_range=255)

def main():
    # Configuration
    base_dir = "carla_benchmark_results/genex_results/genex_1024_unik3d"
    video_name = "087_1_segment_05" # You can change this to other sequences 076_2_segment_09 
    
    # Mask out bottom 20% of the first frame to avoid projecting people
    mask_bottom_ratio = 0.2
    
    npy_path = os.path.join(base_dir, f"{video_name}_distance.npy")
    pose_path = os.path.join(base_dir, f"{video_name}_poses.json")
    video_path = os.path.join(base_dir, f"{video_name}_vis.mp4")
    
    print(f"--- Evaluating {video_name} with bottom {mask_bottom_ratio*100:.0f}% masked ---")
    if not os.path.exists(npy_path):
        print(f"Error: File not found: {npy_path}")
        return

    # Load data
    depths = np.load(npy_path) # [T, H, W]
    with open(pose_path, 'r') as f:
        poses = np.array(json.load(f)) # [T, 4, 4]
        
    vcap = cv2.VideoCapture(video_path)
    video_frames = []
    print("Loading video frames...")
    while True:
        ret, frame = vcap.read()
        if not ret: break
        h_total, w_total = frame.shape[:2]
        # Convert BGR to RGB and create a contiguous copy to avoid negative stride
        video_frames.append(frame[:h_total//2, :, ::-1].copy())
    vcap.release()
    print(f"Loaded {len(video_frames)} frames.")
    
    T, H, W = depths.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate eval_steps to cover more frames, including the last frame
    max_step = T - 1
    eval_steps = [1, 5, 10, 20, 30, 50, max_step]
    # Remove duplicates and steps beyond video length
    eval_steps = sorted(list(set([s for s in eval_steps if s < T])))
    print(f"Will evaluate at steps: {eval_steps}")
    
    # Base frame (usually frame 0)
    idx0 = 0
    img0_torch = torch.from_numpy(video_frames[idx0]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    depth0 = torch.from_numpy(depths[idx0]).unsqueeze(0).unsqueeze(0).to(device)
    pose0 = torch.from_numpy(poses[idx0]).float().to(device)
    
    # 1. Unproject frame 0 to 3D Camera Space
    xyz0 = panorama_to_xyz(depth0) # [1, 3, H, W]
    
    # Create mask for frame 0: mask out bottom mask_bottom_ratio% of the image
    # Bottom 20% means the last 20% rows (highest row indices)
    frame0_mask = torch.ones(1, 1, H, W, device=device)
    bottom_start_row = int(H * (1 - mask_bottom_ratio))
    frame0_mask[:, :, bottom_start_row:, :] = 0  # Mask out bottom region
    print(f"Masking frame 0 from row {bottom_start_row} to {H} (bottom {mask_bottom_ratio*100:.0f}%)")
    
    # Apply mask to frame 0 image - set masked region to black (0)
    # This ensures grid_sample won't sample from the bottom region
    img0_torch_masked = img0_torch * frame0_mask
    
    summary = []

    for step in eval_steps:
        idx1 = idx0 + step
        if idx1 >= T: continue
        
        img1_torch = torch.from_numpy(video_frames[idx1]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        depth1 = torch.from_numpy(depths[idx1]).unsqueeze(0).unsqueeze(0).to(device)
        pose1 = torch.from_numpy(poses[idx1]).float().to(device)
        
        # Backward warping: for each target pixel, sample from source
        # 1. Unproject target frame (frame 1) to 3D using its depth
        xyz1 = panorama_to_xyz(depth1)  # [1, 3, H, W] in frame 1 camera coords
        
        # 2. Transform xyz1 to frame 0's camera space (inverse of forward transform)
        # T_1_to_0 = pose0^-1 * pose1
        rel_pose_inv = torch.inverse(pose0) @ pose1
        
        xyz1_flat = xyz1.view(1, 3, -1)
        xyz1_homo = torch.cat([xyz1_flat, torch.ones(1, 1, H*W, device=device)], dim=1)
        xyz0_from1_homo = torch.matmul(rel_pose_inv, xyz1_homo)
        xyz0_from1 = xyz0_from1_homo[:, :3, :].view(1, 3, H, W)
        
        # 3. Project to frame 0 pixels to get sampling grid
        # This grid tells us: for each pixel in frame 1, where to sample in frame 0
        grid_backward = xyz_to_panorama(xyz0_from1)  # [1, H, W, 2]
        
        # 4. Sample from frame 0 using backward grid (fast!)
        # Use the masked version of frame 0 so bottom region pixels are not sampled
        img0_warped = F.grid_sample(img0_torch_masked, grid_backward, mode='bilinear', 
                                     padding_mode='zeros', align_corners=True)
        
        # Also sample the mask to know which pixels in the warped image come from masked region
        frame0_mask_warped = F.grid_sample(frame0_mask, grid_backward, mode='nearest', 
                                           padding_mode='zeros', align_corners=True)
        
        # 5. Compute valid mask: grid within bounds, valid depth, AND not from masked region
        # For panoramic camera, depth > 0 is always valid (360° view)
        radial_dist = torch.sqrt(xyz0_from1[:, 0]**2 + xyz0_from1[:, 1]**2 + xyz0_from1[:, 2]**2)
        valid_grid = (grid_backward[..., 0] >= -1) & (grid_backward[..., 0] <= 1) & \
                     (grid_backward[..., 1] >= -1) & (grid_backward[..., 1] <= 1)
        valid_depth = radial_dist > 1e-6
        # Add condition: pixel was not sampled from masked region of frame 0
        valid_source = frame0_mask_warped.squeeze(1) > 0.5
        
        mask = valid_grid & valid_depth.squeeze(1) & valid_source
        mask_np = mask.float().squeeze(0).cpu().numpy()
        
        # Convert to numpy for metric
        img1_np = (img1_torch.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img0_warped_np = (img0_warped.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        ssim_val = calculate_ssim(img1_np, img0_warped_np, mask_np)
        
        # Calculate L1 Error in masked area
        l1_diff = np.abs(img1_np.astype(float) - img0_warped_np.astype(float)) / 255.0
        l1_val = np.mean(l1_diff[mask_np > 0.5]) if np.sum(mask_np) > 0 else 1.0
        
        # Calculate coverage (how much of frame 1 can be covered by projecting frame 0)
        coverage = np.sum(mask_np > 0.5) / (H * W) * 100
        
        print(f"Step {step:2d} (Frame {idx1:2d}): SSIM = {ssim_val:.4f}, Masked L1 = {l1_val:.4f}, Coverage = {coverage:.1f}%")
        summary.append(ssim_val)
        
        # Save visualization - vertical stack (top: target frame, bottom: rendered frame from frame 0)
        vis_path = f"eval_{video_name}_masked_step{step}.png"
        
        # Create warped image with holes highlighted
        img0_warped_with_holes = img0_warped_np.copy()
        # Mark invalid regions (holes) in gray for better visualization in papers
        img0_warped_with_holes[mask_np < 0.5] = [128, 128, 128]  # Gray for holes
        
        # Add text labels
        img1_labeled = img1_np.copy()
        img0_warped_labeled = img0_warped_with_holes.copy()
        
        # Add label on images
        cv2.putText(img1_labeled, f"Target Frame {idx1} (Ground Truth)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img0_warped_labeled, f"Frame {idx0} projected to Frame {idx1} (step={step}, bottom {mask_bottom_ratio*100:.0f}% masked)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img0_warped_labeled, f"SSIM={ssim_val:.4f}, L1={l1_val:.4f}, Cov={coverage:.1f}%", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Stack vertically
        combined = np.vstack([img1_labeled, img0_warped_labeled])
        cv2.imwrite(vis_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"  -> Saved visualization to {vis_path}")

    if summary:
        print(f"\nAverage SSIM for {len(summary)} offsets: {np.mean(summary):.4f}")

if __name__ == "__main__":
    main()
