#!/bin/bash

# 默认输入路径
# DEFAULT_NPY="carla_benchmark_results/vipe_pano_zeroshot/town0210_1024_dynamic_fps20_len110/town10_path1_clip_0_seg8_distance.npy"
DEFAULT_NPY="carla_benchmark_results/vipe_pano_pvdepth/town0210_1024_dynamic_fps20_len110/town10_path1_clip_0_seg8_distance.npy"


# 如果命令行提供了参数，则使用参数，否则使用默认值
NPY_PATH=${1:-$DEFAULT_NPY}

# 可选：如果你想指定视频路径也可以加上
# VIDEO_PATH=${2:-""}

echo "Starting visualization for: $NPY_PATH"

if [ ! -f "$NPY_PATH" ]; then
    echo "Error: File $NPY_PATH not found."
    exit 1
fi

python visualize_pano_viser.py \
    --npy_path "$NPY_PATH" \
    --port 8082
    # --video_path "$VIDEO_PATH" # 如果有RGB视频可以取消注释
