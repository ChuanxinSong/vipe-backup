#!/bin/bash

OUTPUT_ROOT_DIR="carla_benchmark_results/vipe_pano_unik3d"
# OUTPUT_ROOT_DIR="carla_benchmark_results/vipe_pano_pvdepth_unik3d"

# 数据集根目录
IMAGE_BASE_DIR="/workspace1/songcx/dataset/pvdepth" 
RESOLUTION=1024

# 运行推理脚本，指定深度方法为 unik3d
# 建议显式设置 CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=6 python run_vipe_pano_all.py \
    --output_root_dir "${OUTPUT_ROOT_DIR}" \
    --resolution "${RESOLUTION}" \
    --image_base_dir "${IMAGE_BASE_DIR}" \
    --depth_method "unik3d"
