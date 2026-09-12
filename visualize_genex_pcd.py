import os
import json
import numpy as np
import torch
import viser
import time
import cv2
import argparse
from pathlib import Path

# --- 绕过 viser 的 psutil 检查错误 ---
try:
    import viser._client_autobuild as _autobuild
    _autobuild.ensure_client_is_built = lambda: None
except ImportError:
    pass
# -----------------------------------

def unproject_panorama(depth, pose, resolution):
    """
    Unproject an equirectangular depth map to 3D points.
    depth: [H, W]
    pose: [4, 4]
    resolution: (W, H)
    """
    H, W = depth.shape
    
    # Create grid of theta and phi
    # Theta (longitude): [0, 2*pi], Phi (latitude): [0, pi]
    theta = np.linspace(0, 2 * np.pi, W, endpoint=False)
    phi = np.linspace(0, np.pi, H)
    
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Convert spherical coordinates to Cartesian coordinates (Local Camera Space)
    # Standard equirectangular mapping:
    # x = sin(phi) * cos(theta)
    # y = sin(phi) * sin(theta)
    # z = cos(phi)
    # We use Vipe/SLAM convention: 
    # Forward is usually Z, but equirectangular starts from a specific angle.
    # Most common: x = depth * sin(phi) * sin(theta), y = depth * cos(phi), z = depth * sin(phi) * cos(theta)
    
    # Vipe Equirectangular projection often follows:
    # x = sin(phi) * sin(theta)
    # y = -cos(phi)
    # z = sin(phi) * cos(theta)
    
    x = depth * np.sin(phi_grid) * np.sin(theta_grid)
    y = -depth * np.cos(phi_grid)
    z = depth * np.sin(phi_grid) * np.cos(theta_grid)
    
    points_local = np.stack([x, y, z, np.ones_like(x)], axis=-1) # [H, W, 4]
    points_local = points_local.reshape(-1, 4)
    
    # Transform to world space
    points_world = points_local @ pose.T # Row major multiplication
    
    return points_world[:, :3]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    # Paths
    base_dir = "carla_benchmark_results/genex_results/genex_1024_unik3d"
    video_name = "076_2_segment_09" # 077_5_segment_07
    npy_path = os.path.join(base_dir, f"{video_name}_distance.npy")
    pose_path = os.path.join(base_dir, f"{video_name}_poses.json")
    
    # Load data
    print(f"Loading depth from {npy_path}...")
    depths = np.load(npy_path) # [T, H, W]
    
    print(f"Loading poses from {pose_path}...")
    with open(pose_path, 'r') as f:
        poses = np.array(json.load(f)) # [T, 4, 4]
        
    # New: Load video for colors
    video_path = os.path.join(base_dir, f"{video_name}_vis.mp4")
    print(f"Loading colors from {video_path}...")
    vcap = cv2.VideoCapture(video_path)
    if not vcap.isOpened():
        print(f"Error: Could not open video {video_path}")
        video_frames = None
    else:
        video_frames = []
        while True:
            ret, frame = vcap.read()
            if not ret:
                break
            # Video is [2H, W, 3], top half is RGB
            h_total, w_total, _ = frame.shape
            rgb = frame[:h_total//2, :, ::-1] # [H, W, 3], BGR to RGB
            video_frames.append(rgb)
        vcap.release()
        print(f"Loaded {len(video_frames)} frames for colors.")

    T, H, W = depths.shape
    print(f"Total frames: {T}, resolution: {W}x{H}")

    # Start Viser server
    server = viser.ViserServer(port=args.port)
    print(f"Viser server started at http://localhost:{args.port}")

    # GUI Elements
    gui_min_depth = server.gui.add_slider("Min Depth", min=0.0, max=10.0, step=0.1, initial_value=0.1)
    gui_max_depth = server.gui.add_slider("Max Depth", min=5.0, max=100.0, step=1.0, initial_value=50.0)
    gui_stride = server.gui.add_slider("Frame Stride", min=1, max=20, step=1, initial_value=5)
    gui_downsample = server.gui.add_slider("Downsample", min=1, max=8, step=1, initial_value=4)
    btn_update = server.gui.add_button("Update Point Cloud")

    # Store data for quick access
    def update_pcd():
        stride = gui_stride.value
        downsample = gui_downsample.value
        min_d = gui_min_depth.value
        max_d = gui_max_depth.value
        
        print(f"Updating point cloud: min={min_d}, max={max_d}, stride={stride}...")
        
        all_pts = []
        all_cols = []
        
        for t in range(0, T, stride):
            depth = depths[t][::downsample, ::downsample]
            curr_H, curr_W = depth.shape # 获取切片后的实际形状
            pose = poses[t]
            
            mask = (depth > min_d) & (depth < max_d)
            if not np.any(mask):
                continue
                
            pts = unproject_panorama(depth, pose, (curr_W, curr_H))
            pts = pts.reshape((curr_H, curr_W, 3))
            
            if video_frames is not None and t < len(video_frames):
                frame_rgb = video_frames[t]
                if frame_rgb.shape[0] != curr_H or frame_rgb.shape[1] != curr_W:
                    frame_rgb = cv2.resize(frame_rgb, (curr_W, curr_H), interpolation=cv2.INTER_AREA)
                cols = frame_rgb[mask]
            else:
                cols = np.ones((pts[mask].shape[0], 3)) * 200
                
            all_pts.append(pts[mask])
            all_cols.append(cols)

        if not all_pts:
            print("No points to display.")
            return

        combined_pts = np.concatenate(all_pts, axis=0)
        combined_cols = np.concatenate(all_cols, axis=0)
        
        server.scene.add_point_cloud(
            "/points",
            points=combined_pts,
            colors=combined_cols,
            point_size=0.01,
        )
        print(f"Done. Total points: {combined_pts.shape[0]}")

    # Initial call
    update_pcd()

    # Register callbacks
    @gui_min_depth.on_update
    def _(_): update_pcd()
    @gui_max_depth.on_update
    def _(_): update_pcd()
    @btn_update.on_click
    def _(_): update_pcd()

    # --- Add Trajectory Visualization ---
    # Extract camera positions from poses [T, 4, 4]
    # In standard SE3 matrix [R|t], the translation is pose[:3, 3]
    cam_positions = poses[:, :3, 3]
    
    # 1. Add a line strip showing the path
    server.scene.add_spline_catmull_rom(
        "/trajectory/path",
        positions=cam_positions,
        color=(255, 0, 0), # Red path
        line_width=2.0,
    )
    
    # 2. Add points at each camera center
    server.scene.add_point_cloud(
        "/trajectory/centers",
        points=cam_positions,
        colors=np.tile(np.array([255, 255, 0]), (cam_positions.shape[0], 1)), # Yellow dots
        point_size=0.03,
    )

    # 3. Add coordinate frames for a subset of poses to show orientation
    for i in range(0, T, 10): # Show orientation every 10 frames
        server.scene.add_frame(
            f"/trajectory/frames/{i}",
            axes_length=0.2,
            axes_radius=0.005,
            wxyz=viser.transforms.SO3.from_matrix(poses[i, :3, :3]).wxyz,
            position=poses[i, :3, 3],
        )

    # Add coordinate frame
    server.scene.add_frame("/world", axes_length=1.0, axes_radius=0.01)

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
