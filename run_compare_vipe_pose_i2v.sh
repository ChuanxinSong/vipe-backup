#!/bin/bash
set -euo pipefail

INPUT_ROOT="${INPUT_ROOT:-}"
SCOPE="${SCOPE:-first_segment}" # first_segment | all_segments
SPLIT_JSON="${SPLIT_JSON:-OmniRoam/configs/train_test_files.json}"
SPLIT_SUBSET="${SPLIT_SUBSET:-test}"
INTERIORGS_ROOT="${INTERIORGS_ROOT:-/data1/songcx/dataset/interiogs_render}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${INPUT_ROOT}/vipe_pose_eval/${SCOPE}}"

FRAMES_SUBDIR="${FRAMES_SUBDIR:-pano_camera0}"
FRAME_EXT="${FRAME_EXT:-png}"
ALIGN_MODE="${ALIGN_MODE:-sim3}"
ROTATION_MODE="${ROTATION_MODE:-constant_offset}"
GT_POSITION_SOURCE="${GT_POSITION_SOURCE:-location}"
GT_ROTATION_SOURCE="${GT_ROTATION_SOURCE:-interiorgs_erp}"
WORLDSCORE_MODE="${WORLDSCORE_MODE:-relative}"
WORLDSCORE_SCALE_SOLVER="${WORLDSCORE_SCALE_SOLVER:-scipy}"
PREVIEW_ROWS="${PREVIEW_ROWS:-0}"

if [[ -z "${INPUT_ROOT}" ]]; then
    echo "[ERROR] INPUT_ROOT is required." >&2
    exit 2
fi
if [[ "${SCOPE}" != "first_segment" && "${SCOPE}" != "all_segments" ]]; then
    echo "[ERROR] SCOPE must be first_segment or all_segments, got: ${SCOPE}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"
LOG_PATH="${OUTPUT_ROOT}/logs/compare.log"
echo "[INFO] poses=${OUTPUT_ROOT}/poses scope=${SCOPE} GT=${INTERIORGS_ROOT}"
echo "[INFO] WorldScore mode=${WORLDSCORE_MODE} scale_solver=${WORLDSCORE_SCALE_SOLVER}"

python -u evaluate_vipe_pose_i2v.py \
    --input_root "${INPUT_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --scope "${SCOPE}" \
    --split_json "${SPLIT_JSON}" \
    --split_subset "${SPLIT_SUBSET}" \
    --interiorgs_root "${INTERIORGS_ROOT}" \
    --frames_subdir "${FRAMES_SUBDIR}" \
    --frame_ext "${FRAME_EXT}" \
    --align_mode "${ALIGN_MODE}" \
    --rotation_mode "${ROTATION_MODE}" \
    --gt_position_source "${GT_POSITION_SOURCE}" \
    --gt_rotation_source "${GT_ROTATION_SOURCE}" \
    --worldscore_mode "${WORLDSCORE_MODE}" \
    --worldscore_scale_solver "${WORLDSCORE_SCALE_SOLVER}" \
    --preview_rows "${PREVIEW_ROWS}" \
    "$@" 2>&1 | tee "${LOG_PATH}"
