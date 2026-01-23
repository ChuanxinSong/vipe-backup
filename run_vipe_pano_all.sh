#!/bin/bash

OUTPUT_ROOT_DIR="carla_benchmark_results/vipe_pano_zeroshot"

# 数据集根目录，请确保与 JSON 中的相对路径能拼接正确
IMAGE_BASE_DIR="/workspace1/songcx/dataset/pvdepth" 
RESOLUTION=1024

# 运行推理脚本
# 建议显式设置 CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=5 python run_vipe_pano_all.py \
    --output_root_dir "${OUTPUT_ROOT_DIR}" \
    --resolution "${RESOLUTION}" \
    --image_base_dir "${IMAGE_BASE_DIR}"