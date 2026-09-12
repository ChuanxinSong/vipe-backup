import numpy as np
import cv2
import argparse
import os
import viser
import viser.transforms as tf
import time
from pathlib import Path

def panorama_to_points(depth_map, rgb_img=None, max_depth=50.0, stride=2):
    """Convert equirectangular depth map to 3D points."""
    H, W = depth_map.shape
    u = np.linspace(0, 1, W)[::stride]
    v = np.linspace(0, 1, H)[::stride]
    uu, vv = np.meshgrid(u, v)
    
    d = depth_map[::stride, ::stride]
    if rgb_img is not None:
        colors = rgb_img[::stride, ::stride].reshape(-1, 3) / 255.0
    else:
        colors = np.ones((d.size, 3)) * 0.5
        
    mask = (d > 0.1) & (d < max_depth)
    d_masked = d[mask]
    uu_masked = uu[mask]
    vv_masked = vv[mask]
    colors_masked = colors[mask.reshape(-1)]
        
    theta = (uu_masked - 0.5) * 2 * np.pi
    phi = vv_masked * np.pi
    
    x = d_masked * np.sin(phi) * np.sin(theta)
    y = -d_masked * np.cos(phi)
    z = d_masked * np.sin(phi) * np.cos(theta)
    
    points = np.stack([x, y, z], axis=-1)
    return points, colors_masked

def main():
    parser = argparse.ArgumentParser(description="Interactive Panorama 3D Visualizer")
    parser.add_argument("--npy_path", type=str, required=True, help="Path to distance .npy file")
    parser.add_argument("--video_path", type=str, help="Path to .mp4 file (RGB top half)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading depths from {args.npy_path}...")
    depths = np.load(args.npy_path)
    num_frames = depths.shape[0] if len(depths.shape) == 3 else 1
    
    video_path = args.video_path
    if not video_path:
        potential = args.npy_path.replace("_distance.npy", "_vis.mp4").replace("_disparity.npy", "_vis.mp4")
        if os.path.exists(potential):
            video_path = potential

# 2. Setup Viser
    # Monkey-patch viser to skip its problematic system process scan
    try:
        import viser._client_autobuild
        viser._client_autobuild._check_viser_yarn_running = lambda: False
        print("Successfully bypassed viser system process scan.")
    except Exception:
        pass

    print(f"Starting Viser server on port {args.port}...")
    try:
        server = viser.ViserServer(port=args.port)
    except Exception as e:
        print(f"Failed to start Viser server: {e}")
        return

    print(f"Viser server successfully started at http://localhost:{args.port}")

    # GUI Controls
    with server.gui.add_folder("Settings"):
        gui_max_depth = server.gui.add_slider("Max Depth", min=1.0, max=100.0, step=1.0, initial_value=50.0)
        gui_stride = server.gui.add_slider("Stride", min=1, max=8, step=1, initial_value=2)
        gui_point_size = server.gui.add_slider("Point Size", min=0.01, max=0.5, step=0.01, initial_value=0.05)
    
    gui_timeline = server.gui.add_slider("Frame Index", min=0, max=num_frames - 1, step=1, initial_value=0)

    # State
    state = {"current_frame": -1}

    def update_frame(_=None):
        idx = int(gui_timeline.value)
        # Check if we need to update
        if idx == state["current_frame"] and not any([gui_max_depth.on_update, gui_stride.on_update]):
            return

        # Load depth
        depth = depths[idx] if len(depths.shape) == 3 else depths
        
        # Load RGB
        rgb = None
        if video_path:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame[0:h//2, :, :], cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != depth.shape:
                    rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
            cap.release()

        # Compute points
        pts, cls = panorama_to_points(depth, rgb, max_depth=gui_max_depth.value, stride=gui_stride.value)
        
        # Update viser
        server.scene.add_point_cloud(
            f"/points",
            points=pts,
            colors=cls,
            point_size=gui_point_size.value,
        )
        state["current_frame"] = idx

    # Callbacks
    @gui_timeline.on_update
    def _(_): update_frame()
    
    @gui_max_depth.on_update
    def _(_): update_frame()
    
    @gui_stride.on_update
    def _(_): update_frame()
    
    @gui_point_size.on_update
    def _(_):
        # Point size can be updated without re-calculating points if we kept the handle,
        # but for simplicity we just re-run update_frame
        update_frame()

    # Initial frame
    update_frame()

    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
