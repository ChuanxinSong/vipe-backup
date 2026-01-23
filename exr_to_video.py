import cv2
import OpenEXR
import Imath
import numpy as np
import glob
import os
from tqdm import tqdm

def load_exr_depth(exr_path):
    """从EXR文件中加载深度数据。"""
    exr_file = OpenEXR.InputFile(exr_path)
    
    # 获取图像尺寸
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # # 通常深度数据存储在'R'（红色）通道中
    # # 读取为32位浮点数
    # pt = Imath.PixelType(Imath.PixelType.FLOAT)
    # red_channel_bytes = exr_file.channel('R', pt)
    
    # # 将字节数据转换为numpy数组
    # depth = np.frombuffer(red_channel_bytes, dtype=np.float32)
    # depth = depth.reshape(height, width)

    # 从 'Z' 通道读取深度信息，这与 DepthCrafter 的习惯一致
    depth_str = exr_file.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth.shape = (height, width)
    depth = depth.copy()
    
    return depth

def visualize_depth(depth_map):
    """将深度图转换为可供观看的彩色图像。"""
    # 替换无穷大和NaN值为0，避免计算错误
    # depth_map[np.isinf(depth_map)] = 0
    # depth_map[np.isnan(depth_map)] = 0

    # # 为了更好的可视化效果，我们忽略纯黑（背景）或过远的点
    # valid_mask = depth_map > 0
    # if valid_mask.any():
    #     min_val = depth_map[valid_mask].min()
    #     max_val = depth_map[valid_mask].max()
    # else:
    #     min_val, max_val = 0, 1 # 如果没有有效深度，则使用默认范围

    # valid_mask = depth_map > 0
    # if valid_mask.any():
        # min_val = depth_map[valid_mask].min()
        # max_val = depth_map[valid_mask].max()

    min_val = depth_map.min()
    max_val = depth_map.max()
    # else:
        # min_val, max_val = 0, 1 # 如果没有有效深度，则使用默认范围

    # 归一化深度值到 0-255 范围
    normalized_depth = (depth_map - min_val) / max(max_val - min_val, 1e-6)
    # normalized_depth[~valid_mask] = 0 # 将无效区域设为0
    
    # 转换为8位整数
    depth_uint8 = (normalized_depth * 255).astype(np.uint8)
    
    # 应用伪彩色图以增强可视化效果
    colored_depth = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    
    # 将无效区域设为黑色
    # colored_depth[~valid_mask] = 0

    return colored_depth

def create_video_from_exrs(exr_dir, output_video_path, fps=30):
    """从EXR文件序列创建视频。"""
    # 查找所有.exr文件并按文件名排序
    exr_files = sorted(glob.glob(os.path.join(exr_dir, '*.exr')))
    
    if not exr_files:
        print(f"错误：在目录 '{exr_dir}' 中未找到.exr文件。")
        return

    # 从第一张图像获取视频尺寸
    first_depth_map = load_exr_depth(exr_files[0])
    height, width = first_depth_map.shape
    
    # 初始化视频写入器
    # 使用 'mp4v' 编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"正在将 {len(exr_files)} 帧图像合成为视频: {output_video_path}")
    
    # 遍历所有exr文件
    for exr_path in tqdm(exr_files, desc="处理帧"):
        depth_map = load_exr_depth(exr_path)
        visual_frame = visualize_depth(depth_map)
        video_writer.write(visual_frame)
        
    # 释放资源
    video_writer.release()
    print("视频生成完毕！")


if __name__ == '__main__':
    # --- 请在这里修改您的路径 ---
    # 包含.exr文件的文件夹路径
    exr_directory = '/home/user/songcx/code/vipe/vipe_results/depth' 
    
    # 输出视频的保存路径和文件名
    output_video_path = '/home/user/songcx/code/vipe/vipe_results/depth_visualization.mp4'
    
    # 视频的帧率
    video_fps = 20
    
    create_video_from_exrs(exr_directory, output_video_path, fps=video_fps)