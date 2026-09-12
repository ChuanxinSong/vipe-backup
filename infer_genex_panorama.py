import os
import time
import torch
import json
import numpy as np
import cv2
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import gc
from omegaconf import OmegaConf

# --- Vipe Imports ---
from vipe.streams.base import VideoFrame, CachedVideoStream, ProcessedVideoStream, CameraType
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.pipeline.processors import EquirectProjectionProcessor
from vipe.pipeline.panorama import MergedPanoramaVideoStream
from vipe.slam.system import SLAMSystem
from vipe.utils.geometry import se3_to_so3, so3_to_se3
from vipe.ext import lietorch as lt

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infer_genex")

# --- 可视化工具 ---
try:
    from depth_eval.utils import save_video
except ImportError:
    def save_video(frames, path, fps=10):
        if len(frames) == 0: return
        h, w, c = frames[0].shape
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in frames:
            out.write((f * 255).astype(np.uint8)[:, :, ::-1])
        out.release()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def colorize_depth_sequence(depths, valid_masks):
    vis = []
    invalid_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    for depth, valid_mask in zip(depths, valid_masks):
        depth_vis = np.full(depth.shape + (3,), invalid_color, dtype=np.float32)
        if np.any(valid_mask):
            disparity = np.zeros_like(depth, dtype=np.float32)
            disparity[valid_mask] = 1.0 / np.maximum(depth[valid_mask], 1e-6)
            disp_valid = disparity[valid_mask]
            disp_min = float(disp_valid.min())
            disp_max = float(disp_valid.max())
            if disp_max - disp_min < 1e-8:
                disp_norm = np.zeros_like(disparity, dtype=np.float32)
            else:
                disp_norm = np.zeros_like(disparity, dtype=np.float32)
                disp_norm[valid_mask] = (disparity[valid_mask] - disp_min) / (disp_max - disp_min)

            disp_color = cv2.applyColorMap((disp_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            disp_color = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            depth_vis[valid_mask] = disp_color[valid_mask]
        vis.append(depth_vis)

    return np.stack(vis)

# --- 适配器：支持缩放的视频流 ---
class ResizedRawMp4Stream(RawMp4Stream):
    def __init__(self, path, resolution=(1024, 512), **kwargs):
        super().__init__(Path(path), **kwargs)
        self.target_width = resolution[0]
        self.target_height = resolution[1]

    def frame_size(self) -> tuple[int, int]:
        return (self.target_height, self.target_width)

    def __next__(self) -> VideoFrame:
        frame = super().__next__()
        # frame.rgb is [H, W, 3] on CUDA
        rgb = frame.rgb.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        rgb = torch.nn.functional.interpolate(rgb, size=(self.target_height, self.target_width), mode='area')
        frame.rgb = rgb.squeeze(0).permute(1, 2, 0)
        frame.camera_type = CameraType.PANORAMA
        return frame

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

def run_genex_inference(device, args):
    slam_cfg_default, virtual_cfg_default = load_vipe_configs()

    with open(args.json_path, 'r') as f:
        video_paths = json.load(f)
    
    print(f"Loaded {len(video_paths)} videos.")
    
    for i, video_path_full in enumerate(video_paths):
        video_name = os.path.basename(video_path_full).split('.')[0]
        print(f"\n=== Processing Video {i+1}/{len(video_paths)}: {video_name} ===")
        start_time = time.time()
        
        # 输出路径设定
        out_dir = os.path.join(args.output_root_dir, f"genex_{args.resolution}_{args.depth_method}")
        os.makedirs(out_dir, exist_ok=True)
        npy_path = os.path.join(out_dir, f"{video_name}_distance.npy")
        valid_mask_path = os.path.join(out_dir, f"{video_name}_distance_valid_mask.npy")
        pose_path = os.path.join(out_dir, f"{video_name}_poses.json") # 改为 json 格式
        vis_path = os.path.join(out_dir, f"{video_name}_vis.mp4")

        if os.path.exists(npy_path) and os.path.exists(valid_mask_path) and os.path.exists(pose_path):
            print("Result exists, skipping.")
            continue
            
        # A. 创建视频流
        video_stream = ResizedRawMp4Stream(video_path_full, resolution=(args.resolution, args.resolution // 2))
        cached_video_stream = CachedVideoStream(video_stream, desc="Caching Video")
        
        # B. 准备虚拟相机 Rig
        virtual_cfg = virtual_cfg_default
        pano_height = args.resolution // 2
        virtual_height = pano_height // 2
        virtual_height = virtual_height + (virtual_height % 2)
        virtual_focal = virtual_height / (2 * np.tan(np.deg2rad(virtual_cfg.fovx) / 2))
        virtual_width = int(virtual_focal * np.tan(np.deg2rad(virtual_cfg.fovx) / 2) * 2)
        virtual_width = virtual_width + (virtual_width % 2)
        
        virtual_intrinsics = (
            torch.tensor([virtual_focal, virtual_focal, virtual_width // 2, virtual_height // 2])
            .float().to(device)
        )
        virtual_size = (virtual_height, virtual_width)
        
        rig_transforms = [
            so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(yaw, 0.0))
            for yaw in np.linspace(0, 2 * np.pi, virtual_cfg.num_views, endpoint=False)
        ]
        if virtual_cfg.top:
            rig_transforms.append(so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(0.0, np.pi / 2)))
        if virtual_cfg.bottom:
            rig_transforms.append(so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(0.0, -np.pi / 2)))
            
        # C. 构建投影流
        slam_streams = []
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
        
        # D. 运行 SLAM
        print(" -> Running SLAM...")
        slam_pipeline = SLAMSystem(device=device, config=slam_cfg_default)
        slam_output = slam_pipeline.run(slam_streams, rig=rig_se3)
        
        # 保存 SLAM 位姿 (Trajectory)
        # slam_output.trajectory 是一个 SE3 对象，我们可以转换成 [T, 4, 4] 的 numpy 矩阵
        poses = slam_output.trajectory.matrix().cpu().numpy() # [T, 4, 4]
        # 转换为 list 格式并保存为 JSON，更方便直接查看
        with open(pose_path, 'w') as f:
            json.dump(poses.tolist(), f, indent=2)
        print(f" -> Saved Poses: {pose_path}")
        
        # E. 深度融合
        print(f" -> Merging Panorama Depth (Vipe + {args.depth_method})...")
        depth_options = {
            "unet_path": args.unet_path,
            "ppl_type": args.ppl_type,
            "num_denoising_steps": args.num_denoising_steps,
            "guidance_scale": args.guidance_scale,
            "cpu_offload": args.cpu_offload,
            "slam_infill": args.slam_infill,
        }
        
        output_stream = MergedPanoramaVideoStream(
            cached_video_stream,
            slam_output,
            pano_depth_method=args.depth_method,
            depth_options=depth_options,
        )
        
        # F. 收集结果
        all_distances = []
        all_valid_masks = []
        original_rgbs = []
        
        for frame in tqdm(output_stream, total=len(video_stream), desc="Fusing"):
            depth = frame.metric_depth.cpu().numpy()
            all_distances.append(depth)
            all_valid_masks.append(np.isfinite(depth) & (depth > 0))
            rgb = frame.rgb.cpu().numpy()
            original_rgbs.append((rgb * 255).astype(np.uint8))
            
        # 3. 保存
        all_distances = np.stack(all_distances)
        all_valid_masks = np.stack(all_valid_masks).astype(np.uint8)
        np.save(npy_path, all_distances)
        np.save(valid_mask_path, all_valid_masks)
        print(f" -> Saved NPY: {npy_path}")
        print(f" -> Saved Valid Mask: {valid_mask_path}")

        # 4. 可视化
        print(f" -> Generating video...")
        vis_colored = colorize_depth_sequence(all_distances, all_valid_masks.astype(bool))
        
        combined_frames = []
        for idx in range(len(original_rgbs)):
            rgb_float = original_rgbs[idx].astype(np.float32) / 255.0
            depth_vis = vis_colored[idx]
            combined = np.concatenate([rgb_float, depth_vis], axis=0) # 上下拼接
            combined_frames.append(combined)
            
        save_video(combined_frames, vis_path, fps=10)
        print(f" -> Video saved: {vis_path}")
        
        print(f" === Done in {time.time() - start_time:.2f}s ===")

        del slam_pipeline, slam_output, output_stream, cached_video_stream, slam_streams
        gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default="genex_realworld_top50.json")
    parser.add_argument("--output_root_dir", default="carla_benchmark_results/genex_results")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--depth_method", type=str, default="unik3d", choices=["unik3d", "depthcrafter", "raw_slam"])
    parser.add_argument("--slam_infill", type=str2bool, default=False)
    
    parser.add_argument("--unet_path", type=str, default="tencent/DepthCrafter")
    parser.add_argument("--ppl_type", type=str, default="depthcrafter")
    parser.add_argument("--num_denoising_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--cpu_offload", type=str, default=None)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_genex_inference(device, args)
