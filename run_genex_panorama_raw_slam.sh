#!/bin/bash

# 输出根目录
OUTPUT_ROOT_DIR="carla_benchmark_results/genex_results_raw_slam"

JSON_PATH="genex_realworld_top50.json"
RESOLUTION=1024
DEPTH_METHOD="raw_slam"
SLAM_INFILL=false

# 运行推理脚本
CUDA_VISIBLE_DEVICES=0 python infer_genex_panorama.py \
    --json_path "${JSON_PATH}" \
    --output_root_dir "${OUTPUT_ROOT_DIR}" \
    --resolution "${RESOLUTION}" \
    --depth_method "${DEPTH_METHOD}" \
    --slam_infill "${SLAM_INFILL}"
