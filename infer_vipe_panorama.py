import os
import torch
import json
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
# 确保在 vipe-panorama 根目录下运行，或者 vipe 在 PYTHONPATH 中
from vipe.streams.base import VideoStream, VideoFrame, CachedVideoStream, ProcessedVideoStream, CameraType
from vipe.pipeline.processors import EquirectProjectionProcessor
from vipe.pipeline.panorama import MergedPanoramaVideoStream
from vipe.slam.system import SLAMSystem
from vipe.utils.geometry import se3_to_so3, so3_to_se3
from vipe.ext import lietorch as lt
from vipe.utils import io

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infer_vipe")

# --- 可视化工具 (复用您提供的代码) ---
try:
    from depth_eval.utils import vis_sequence_depth, save_video
except ImportError:
    print("Warning: depth_eval.utils not found. Using fallback visualization.")
    def vis_sequence_depth(depths):
        vis = []
        for d in depths:
            d = 1.0 / (d + 1e-6)
            d_min, d_max = d.min(), d.max()
            d_norm = (d - d_min) / (d_max - d_min + 1e-8)
            d_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            vis.append(cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB) / 255.0)
        return np.array(vis)
    
    def save_video(frames, path, fps=20):
        if len(frames) == 0: return
        h, w, c = frames[0].shape
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in frames:
            out.write((f * 255).astype(np.uint8)[:, :, ::-1])
        out.release()

# --- 适配器：将 JSON 数据转换为 Vipe 的 VideoStream ---
class SimpleJsonVideoStream(VideoStream):
    def __init__(self, image_paths, resolution=(1024, 512), fps=10.0, name="json_stream"):
        self.image_paths = image_paths
        self.width = resolution[0]
        self.height = resolution[1]
        self._fps = fps
        self._name = name

    def frame_size(self) -> tuple[int, int]:
        return (self.height, self.width)
    
    def fps(self) -> float:
        return self._fps
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def name(self) -> str:
        return self._name
    
    def __iter__(self):
        for i, path in enumerate(self.image_paths):
            if not os.path.exists(path):
                logger.error(f"Image not found: {path}")
                # 生成纯黑帧防止崩溃
                img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                img = cv2.imread(path)
                if img is None:
                    logger.error(f"Failed to load image: {path}")
                    img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    # Resize 到指定分辨率 (Vipe 需要统一分辨率)
                    if img.shape[1] != self.width or img.shape[0] != self.height:
                        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
            
            # 转 RGB 并归一化到 0-1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = torch.from_numpy(img).float() / 255.0
            
            yield VideoFrame(
                raw_frame_idx=i,
                rgb=rgb,
                camera_type=CameraType.PANORAMA
            )

# --- 辅助函数 ---
def extract_file_parameters(json_filename):
    setting = "dynamic" if "dynamic" in json_filename else "static"
    fps_str = next((x for x in ["fps01", "fps10", "fps20", "fps02"] if x in json_filename), "fps_unknown")
    len_str = next((x for x in ["len20", "len50", "len90", "len110"] if x in json_filename), "len_unknown")
    return setting, fps_str, len_str

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

# --- 主逻辑 ---
def run_vipe_inference(device, args):
    slam_cfg_default, virtual_cfg_default = load_vipe_configs()

    # 1. 加载数据列表
    with open(args.json_path, 'r') as f:
        clips_raw = json.load(f)
    
    all_clips = []
    for town, t_data in clips_raw.items():
        for path, p_data in t_data.items():
            for c_name, frames in p_data.items():
                all_clips.append({"id": c_name, "frames": frames, "path": path})

    print(f"Loaded {len(all_clips)} clips.")
    
    # 2. 遍历处理
    for i, clip in enumerate(all_clips):
        clip_name = clip["id"]
        print(f"\n=== Processing Clip {i+1}/{len(all_clips)}: {clip_name} ===")
        
        # 准备路径
        img_paths = [os.path.join(args.image_base_dir, f['rgb_path']) for f in clip['frames']]
        
        # 输出目录设置
        setting, fps_str, len_str = extract_file_parameters(args.json_path)
        out_dir = os.path.join(args.output_root_dir, f"town0210_{args.resolution}_{setting}_{fps_str}_{len_str}")
        os.makedirs(out_dir, exist_ok=True)
        npy_path = os.path.join(out_dir, f"{clip_name}_distance.npy")
        video_path = os.path.join(out_dir, f"{clip_name}_vis.mp4")

        if os.path.exists(npy_path):
            print("Result exists, skipping.")
            continue
            
        # ---------------- Vipe Pipeline Logic Start ----------------
        # 这部分逻辑直接复刻自 vipe.pipeline.panorama.PanoramaAnnotationPipeline.run
        
        # A. 创建视频流
        video_stream = SimpleJsonVideoStream(img_paths, resolution=(args.resolution, args.resolution // 2), name=clip_name)
        cached_video_stream = CachedVideoStream(video_stream, desc="Loading Frames")
        
        # B. 准备虚拟相机 Rig (切分全景图)
        # import pdb; pdb.set_trace()
        virtual_cfg = virtual_cfg_default
        pano_height = args.resolution // 2
        virtual_height = pano_height // 2
        virtual_height = virtual_height + (virtual_height % 2) # 确保是偶数
        # virtual_height = virtual_cfg.height
        virtual_focal = virtual_height / (2 * np.tan(np.deg2rad(virtual_cfg.fovx) / 2))
        virtual_width = int(virtual_focal * np.tan(np.deg2rad(virtual_cfg.fovx) / 2) * 2)
        virtual_width = virtual_width + (virtual_width % 2)
        
        virtual_intrinsics = (
            torch.tensor([virtual_focal, virtual_focal, virtual_width // 2, virtual_height // 2])
            .float().cuda()
        )
        virtual_size = (virtual_height, virtual_width)
        
        # 生成 6 面视角 (前后左右上下)
        rig_transforms = [
            so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(yaw, 0.0))
            for yaw in np.linspace(0, 2 * np.pi, virtual_cfg.num_views, endpoint=False)
        ]
        if virtual_cfg.top:
            rig_transforms.append(so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(0.0, np.pi / 2)))
        if virtual_cfg.bottom:
            rig_transforms.append(so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(0.0, -np.pi / 2)))
            
        # C. 构建 SLAM 输入流 (投影处理)
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
        
        # D. 运行 SLAM 系统
        print(" -> Running SLAM...")
        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=slam_cfg_default)
        slam_output = slam_pipeline.run(slam_streams, rig=rig_se3)
        
        # E. 运行深度融合 (MergedPanoramaVideoStream)
        # 这里的关键是 depth_align_model="unik3d"，它会触发内部调用 Unik3D 并与 SLAM 稀疏点对齐
        print(" -> Merging Panorama Depth (Vipe + Unik3D)...")
        output_stream = MergedPanoramaVideoStream(
            cached_video_stream,
            # slam_streams,
            slam_output,
            pano_depth_method="unik3d", # 强制指定 unik3d
        )
        
        # F. 收集结果
        all_distances = []
        original_rgbs = []
        
        for frame in tqdm(output_stream, total=len(video_stream), desc="Fusing"):
            # metric_depth 是融合并对齐后的深度
            depth = frame.metric_depth.cpu().numpy()
            all_distances.append(depth)
            
            # 保存 RGB 用于可视化
            rgb = frame.rgb.cpu().numpy()
            original_rgbs.append((rgb * 255).astype(np.uint8))
            
        # ---------------- Vipe Pipeline Logic End ----------------

        # 3. 保存结果
        all_distances = np.stack(all_distances) # [T, H, W]
        np.save(npy_path, all_distances)
        print(f" -> Saved NPY: {npy_path}")

        # 4. 可视化
        print(f" -> Generating video...")
        disparity = 1.0 / (all_distances + 1e-6)
        disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)
        vis_colored = vis_sequence_depth(disparity)
        
        combined_frames = []
        for idx in range(len(original_rgbs)):
            rgb_float = original_rgbs[idx].astype(np.float32) / 255.0
            depth_vis = vis_colored[idx]
            combined = np.concatenate([rgb_float, depth_vis], axis=0)
            combined_frames.append(combined)
            
        save_video(combined_frames, video_path, fps=10)
        print(f" -> Video saved: {video_path}")
        
        # 清理显存
        del slam_pipeline, slam_output, output_stream, cached_video_stream, slam_streams
        gc.collect()
        # torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Path to the input JSON file defining clips")
    parser.add_argument("--image_base_dir", required=True, help="Base directory for images")
    parser.add_argument("--output_root_dir", default="vipe_results", help="Root directory for outputs")
    parser.add_argument("--resolution", type=int, default=1024, help="Width of the panorama (Height will be Width/2)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing Vipe Inference...")
    
    run_vipe_inference(device, args)