#!/bin/bash

# OUTPUT_ROOT_DIR="carla_benchmark_results/vipe_pano_zeroshot"
OUTPUT_ROOT_DIR="carla_benchmark_results/vipe_pano_pvdepth"


# 数据集根目录，请确保与 JSON 中的相对路径能拼接正确
IMAGE_BASE_DIR="/workspace1/songcx/dataset/pvdepth" 
RESOLUTION=1024

# DepthCrafter 魔改版配置 (参考 DepthCrafter/run_infer_town0210.sh)
PPL_TYPE="distortion_noise_annealed_weighting"
UNET_PATH="/home/user/songcx/code/DepthCrafter/sft_pano_depthcrafter/20260119-082318/sft_res640_spatial_distortion_noise_annealed_10k_weighting_20k_step/unet"

# DepthCrafter 预计算结果目录 (跳过推理阶段，直接使用现成的 npy)
NPY_DIR="DepthCrafter/carla_benchmark_results/sft_res640_spatial_w_cube_temporal_fusion_distortion_noise_annealed_weighting_cubeTempDelay20k_train_only_cube_temp_proj_w_cube_fea_for_kv_20k_step"

# 运行推理脚本
# 建议显式设置 CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=3 python run_vipe_pano_all.py \
    --output_root_dir "${OUTPUT_ROOT_DIR}" \
    --resolution "${RESOLUTION}" \
    --image_base_dir "${IMAGE_BASE_DIR}" \
    --ppl_type "${PPL_TYPE}" \
    --unet_path "${UNET_PATH}" \
    --depthcrafter_npy_dir "${NPY_DIR}"