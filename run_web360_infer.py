import os
import torch
import numpy as np
import cv2
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
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
from vipe.utils import io

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infer_web360")

# --- 可视化工具 (复用您提供的代码) ---
from depth_eval.utils import vis_sequence_depth, save_video

def load_vipe_configs():
    # [Fix] 正确加载并合并配置
    # 1. 加载默认 SLAM 配置 (开启了 keyframe_depth)
    slam_cfg_default = OmegaConf.load("configs/slam/default.yaml")
    
    # 2. 加载全景 Pipeline 配置 (其中 slam 部分将 keyframe_depth 设为了 null)
    pano_pipeline_cfg = OmegaConf.load("configs/pipeline/panorama.yaml")
    
    # 3. 合并配置: 用 pipeline 的设置覆盖 default
    if "slam" in pano_pipeline_cfg:
        slam_cfg = OmegaConf.merge(slam_cfg_default, pano_pipeline_cfg.slam)
    else:
        slam_cfg = slam_cfg_default

    # 4. 强制覆盖一些推理脚本特定的参数
    slam_cfg.visualize = False 
    
    # 确保 optimize_intrinsics 有值 (default 中是 '???')
    if OmegaConf.is_missing(slam_cfg, "optimize_intrinsics"):
         slam_cfg.optimize_intrinsics = False
    
    # [关键] 再次确保 keyframe_depth 为 None，防止合并失败
    slam_cfg.keyframe_depth = None
    
    return slam_cfg, pano_pipeline_cfg.virtual

class Web360Mp4Stream(RawMp4Stream):
    def __init__(self, path: Path, resolution: tuple[int, int] = None, name: str | None = None) -> None:
        super().__init__(path, name=name)
        self.resolution = resolution # (width, height)

    def frame_size(self) -> tuple[int, int]:
        if self.resolution:
            return (self.resolution[1], self.resolution[0])
        return super().frame_size()

    def __next__(self) -> VideoFrame:
        # Note: self.vcap is initialized in RawMp4Stream.__iter__
        while True:
            ret, frame = self.vcap.read()
            self.current_frame_idx += 1

            if not ret:
                self.vcap.release()
                raise StopIteration

            if self.current_frame_idx >= self.end:
                self.vcap.release()
                raise StopIteration

            if self.current_frame_idx < self.start:
                continue

            if (self.current_frame_idx - self.start) % self.step == 0:
                break

        if self.resolution:
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = torch.as_tensor(frame).float() / 255.0
        frame_rgb = frame_rgb.cuda()

        return VideoFrame(
            raw_frame_idx=self.current_frame_idx, 
            rgb=frame_rgb,
            camera_type=CameraType.PANORAMA
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_file", default="web360_infer.txt", help="Path to the list of mp4 files")
    parser.add_argument("--input_root", default="/data3/songcx/dataset/web360/web360_for_depthcrafter/rgb")
    parser.add_argument("--output_root", default="/data3/songcx/results/vipe/web360_results")
    parser.add_argument("--resolution", type=int, default=1024, help="Pano width")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slam_cfg, virtual_cfg = load_vipe_configs()

    if not os.path.exists(args.list_file):
        logger.error(f"List file not found: {args.list_file}")
        return

    with open(args.list_file, 'r') as f:
        video_files = [line.strip() for line in f if line.strip()]

    os.makedirs(args.output_root, exist_ok=True)

    for video_file in video_files:
        video_path = os.path.join(args.input_root, video_file)
        # import pdb; pdb.set_trace()
        video_name = Path(video_file).stem
        out_root_dir = os.path.join(args.output_root, video_name)
        os.makedirs(out_root_dir, exist_ok=True)
        
        npy_path = os.path.join(out_root_dir, f"{video_name}_distance.npy")
        vis_video_path = os.path.join(out_root_dir, f"{video_name}_vis.mp4")
        frame_out_dir = os.path.join(out_root_dir, "frames")
        
        if os.path.exists(npy_path) and os.path.exists(vis_video_path) and os.path.exists(frame_out_dir):
            logger.info(f"Skipping {video_name}, already exists.")
            continue

        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            continue

        logger.info(f"\n>>> Processing {video_name} ...")
        
        # 1. Video Stream
        video_stream = Web360Mp4Stream(Path(video_path), resolution=(args.resolution, args.resolution // 2), name=video_name)
        cached_video_stream = CachedVideoStream(video_stream, desc=f"Loading {video_name}")

        # 2. Virtual Rig Setup
        pano_height = args.resolution // 2
        virtual_height = pano_height // 2
        virtual_height = virtual_height + (virtual_height % 2)
        virtual_focal = virtual_height / (2 * np.tan(np.deg2rad(virtual_cfg.fovx) / 2))
        virtual_width = int(virtual_focal * np.tan(np.deg2rad(virtual_cfg.fovx) / 2) * 2)
        virtual_width = virtual_width + (virtual_width % 2)
        
        virtual_intrinsics = (
            torch.tensor([virtual_focal, virtual_focal, virtual_width // 2, virtual_height // 2])
            .float().cuda()
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
            
        slam_streams = []
        for rig_transform in rig_transforms:
            projectors = [EquirectProjectionProcessor(se3_to_so3(rig_transform), virtual_size, virtual_intrinsics)]
            slam_streams.append(ProcessedVideoStream(cached_video_stream, projectors).cache(online=True))
        
        rig_se3 = lt.stack(rig_transforms, dim=0)

        # 3. SLAM
        slam_pipeline = SLAMSystem(device=device, config=slam_cfg)
        slam_output = slam_pipeline.run(slam_streams, rig=rig_se3)

        # 4. Fusion
        output_stream = MergedPanoramaVideoStream(cached_video_stream, slam_output, pano_depth_method="unik3d")

        all_distances = []
        original_rgbs = []
        
        for frame in tqdm(output_stream, total=len(video_stream), desc="Fusing"):
            # metric_depth 是融合并对齐后的深度
            depth = frame.metric_depth.cpu().numpy()
            all_distances.append(depth)
            
            # 保存 RGB 用于可视化
            rgb = frame.rgb.cpu().numpy()
            original_rgbs.append((rgb * 255).astype(np.uint8))

        # 3. 保存结果
        all_distances = np.stack(all_distances) # [T, H, W]
        np.save(npy_path, all_distances)
        logger.info(f" -> Saved NPY: {npy_path}")

        # 4. 可视化
        logger.info(f" -> Generating video and saving frames...")
        os.makedirs(frame_out_dir, exist_ok=True)
        disparity = 1.0 / (all_distances + 1e-6)
        disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)
        vis_colored = vis_sequence_depth(disparity)
        
        combined_frames = []
        for idx in range(len(original_rgbs)):
            rgb_float = original_rgbs[idx].astype(np.float32) / 255.0
            depth_vis = vis_colored[idx]
            combined = np.concatenate([rgb_float, depth_vis], axis=0) # [2H, W, 3]
            combined_frames.append(combined)
            
            # Save individual frame
            frame_vis_path = os.path.join(frame_out_dir, f"{idx:06d}.png")
            cv2.imwrite(frame_vis_path, cv2.cvtColor((combined * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            
        save_video(combined_frames, vis_video_path, fps=20)
        logger.info(f" -> Video saved: {vis_video_path}")
        logger.info(f" -> Frames saved in: {frame_out_dir}")
        
        logger.info(f"Done: {video_name}")
        
        # 清理显存
        del slam_pipeline, slam_output, output_stream, cached_video_stream, slam_streams
        gc.collect()
        # torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
