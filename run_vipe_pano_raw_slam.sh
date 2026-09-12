#!/bin/bash

OUTPUT_ROOT_DIR="carla_benchmark_results/vipe_pano_raw_slam"

# 数据集根目录
IMAGE_BASE_DIR="/workspace1/songcx/dataset/pvdepth"
RESOLUTION=1024
SLAM_INFILL=false

# 运行推理脚本，指定深度方法为 raw_slam
# 建议显式设置 CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=6 python run_vipe_pano_all.py \
    --output_root_dir "${OUTPUT_ROOT_DIR}" \
    --resolution "${RESOLUTION}" \
    --image_base_dir "${IMAGE_BASE_DIR}" \
    --depth_method "raw_slam" \
    --slam_infill "${SLAM_INFILL}"
