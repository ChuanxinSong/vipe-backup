#!/bin/bash
set -euo pipefail

VIDEO_ID="${VIDEO_ID:-0007_840137}"
VIPE_POSE_ROOT="${VIPE_POSE_ROOT:-carla_benchmark_results/vipe_pano_pose/train_test_files_omniroam_png}"
INTERIORGS_ROOT="${INTERIORGS_ROOT:-/data1/songcx/dataset/interiogs_render}"
OUTPUT_ROOT="${OUTPUT_ROOT:-pose_eval_results/vipe_pano_pose_eval}"

FRAMES_SUBDIR="${FRAMES_SUBDIR:-pano_camera0}"
FRAME_EXT="${FRAME_EXT:-png}"
FRAME_START="${FRAME_START:-1}"
FRAME_COUNT="${FRAME_COUNT:-}"

ALIGN_MODE="${ALIGN_MODE:-sim3}"  # none | scale | sim3
ROTATION_MODE="${ROTATION_MODE:-absolute}"  # absolute | relative | constant_offset
GT_POSITION_SOURCE="${GT_POSITION_SOURCE:-location}"  # location | rt
GT_ROTATION_SOURCE="${GT_ROTATION_SOURCE:-interiorgs_erp}"  # interiorgs_erp | rt | identity
WORLDSCORE_MODE="${WORLDSCORE_MODE:-relative}"  # relative | raw
PREVIEW_ROWS="${PREVIEW_ROWS:-10}"

VIPE_POSES="${VIPE_POSES:-${VIPE_POSE_ROOT}/${VIDEO_ID}_poses.json}"
GT_TRANSFORMS_JSON="${GT_TRANSFORMS_JSON:-${INTERIORGS_ROOT}/${VIDEO_ID}/transforms.json}"

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${VIDEO_ID}}"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/${VIDEO_ID}_vipe_vs_gt.csv}"
OUTPUT_SUMMARY_JSON="${OUTPUT_SUMMARY_JSON:-${OUTPUT_DIR}/${VIDEO_ID}_vipe_vs_gt_summary.json}"

ARGS=(
    --vipe_poses "${VIPE_POSES}"
    --gt_transforms_json "${GT_TRANSFORMS_JSON}"
    --frames_subdir "${FRAMES_SUBDIR}"
    --frame_ext "${FRAME_EXT}"
    --frame_start "${FRAME_START}"
    --align_mode "${ALIGN_MODE}"
    --rotation_mode "${ROTATION_MODE}"
    --gt_position_source "${GT_POSITION_SOURCE}"
    --gt_rotation_source "${GT_ROTATION_SOURCE}"
    --worldscore_mode "${WORLDSCORE_MODE}"
    --output_csv "${OUTPUT_CSV}"
    --output_summary_json "${OUTPUT_SUMMARY_JSON}"
    --preview_rows "${PREVIEW_ROWS}"
)

if [[ -n "${FRAME_COUNT}" ]]; then
    ARGS+=(--frame_count "${FRAME_COUNT}")
fi

python compare_vipe_omniroam_pose.py "${ARGS[@]}" "$@"
