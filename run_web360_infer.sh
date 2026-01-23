#!/bin/bash

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=5

# 输入列表文件
LIST_FILE="web360_infer.txt"

# 数据集根目录
INPUT_ROOT="/data3/songcx/dataset/web360/web360_for_depthcrafter/rgb/web360"

# 输出根目录
OUTPUT_ROOT="/data3/songcx/results/vipe/web360_results"

# 分辨率 (全景图宽度)
RESOLUTION=1024

# 运行推理脚本
python run_web360_infer.py \
    --list_file "${LIST_FILE}" \
    --input_root "${INPUT_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --resolution "${RESOLUTION}"
