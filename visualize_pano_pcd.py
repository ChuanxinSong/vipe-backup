import numpy as np
import cv2
import argparse
import torch
import os
from pathlib import Path

def panorama_to_points(depth_map, rgb_img=None, max_depth=50.0, stride=2):
    """
    Convert an equirectangular depth map to a 3D point cloud.
    
    depth_map: [H, W] depth in meters
    rgb_img: [H, W, 3] optional colors [0, 255]
    max_depth: points further than this are filtered
    stride: downsampling factor for performance
    """
    H, W = depth_map.shape
    
    # 1. Generate UV grid
    u = np.linspace(0, 1, W)[::stride]
    v = np.linspace(0, 1, H)[::stride]
    uu, vv = np.meshgrid(u, v)
    
    # 2. Extract depths and colors at stride
    d = depth_map[::stride, ::stride]
    if rgb_img is not None:
        colors = rgb_img[::stride, ::stride].reshape(-1, 3) / 255.0
    else:
        colors = None
        
    # 3. Mask valid points
    mask = (d > 0.1) & (d < max_depth)
    d = d[mask]
    uu = uu[mask]
    vv = vv[mask]
    if colors is not None:
        colors = colors[mask.reshape(-1)]
        
    # 4. Inverse Projection Formula (Vipe Convention)
    theta = (uu - 0.5) * 2 * np.pi
    phi = vv * np.pi
    
    x = d * np.sin(phi) * np.sin(theta)
    y = -d * np.cos(phi)
    z = d * np.sin(phi) * np.cos(theta)
    
    points = np.stack([x, y, z], axis=-1)
    return points, colors

def save_ply(filename, points, colors=None):
    """Save points as a PLY file."""
    header = """ply
format ascii 1.0
element vertex {0}
property float x
property float y
property float z
{1}
end_header
"""
    color_header = "property uchar red\nproperty uchar green\nproperty uchar blue" if colors is not None else ""
    with open(filename, 'w') as f:
        f.write(header.format(len(points), color_header))
        for i in range(len(points)):
            p = points[i]
            line = f"{p[0]} {p[1]} {p[2]}"
            if colors is not None:
                c = (colors[i] * 255).astype(np.uint8)
                line += f" {c[0]} {c[1]} {c[2]}"
            f.write(line + "\n")

def main():
    parser = argparse.ArgumentParser(description="Visualize Panorama Depth as Point Cloud")
    parser.add_argument("--npy_path", type=str, required=True, help="Path to the distance .npy file")
    parser.add_argument("--img_dir", type=str, help="Directory containing RGB images (optional)")
    parser.add_argument("--video_path", type=str, help="Path to the .mp4 video file containing RGB (top half)")
    parser.add_argument("--frame_idx", type=int, default=0, help="Which frame to visualize")
    parser.add_argument("--max_depth", type=float, default=50.0, help="Max distance to include")
    parser.add_argument("--stride", type=int, default=2, help="Subsampling stride")
    parser.add_argument("--output", type=str, default="output.ply", help="Output filename")
    
    args = parser.parse_args()
    
    # 1. Load Depth
    print(f"Loading depth from {args.npy_path}...")
    depths = np.load(args.npy_path)
    if len(depths.shape) == 3:
        depth = depths[args.frame_idx]
    else:
        depth = depths
        
    # 2. Try to find corresponding RGB
    rgb = None
    
    # 2.1 Auto-detect video if not provided
    video_path = args.video_path
    if not video_path and not args.img_dir:
        # Check if a .mp4 with same prefix exists
        potential_video = args.npy_path.replace("_distance.npy", "_vis.mp4").replace("_disparity.npy", "_vis.mp4")
        if os.path.exists(potential_video):
            video_path = potential_video
            print(f"Auto-detected video: {video_path}")

    if video_path:
        print(f"Loading color from video: {video_path} (frame {args.frame_idx})...")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
        ret, frame = cap.read()
        if ret:
            # Crop top half (RGB)
            h, w, _ = frame.shape
            rgb_half = frame[0:h//2, :, :]
            rgb = cv2.cvtColor(rgb_half, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != depth.shape:
                rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            print(f"Failed to read frame {args.frame_idx} from video.")
        cap.release()
    elif args.img_dir:
        # Simple heuristic to find common image names
        img_files = sorted([f for f in os.listdir(args.img_dir) if f.endswith(('.png', '.jpg'))])
        if args.frame_idx < len(img_files):
            img_path = os.path.join(args.img_dir, img_files[args.frame_idx])
            print(f"Loading color from {img_path}...")
            rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != depth.shape:
                rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
    print("Converting to 3D points...")
    points, colors = panorama_to_points(depth, rgb, max_depth=args.max_depth, stride=args.stride)
    
    # 4. Save
    print(f"Saving {len(points)} points to {args.output}...")
    save_ply(args.output, points, colors)
    print("Done! You can open the .ply file in MeshLab or CloudCompare.")

if __name__ == "__main__":
    main()
