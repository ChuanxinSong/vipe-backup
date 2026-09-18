#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Both scopes intentionally default to the same GPU and run concurrently.
GPU_ID="${GPU_ID:-0}"
GPU_IDS_81="${GPU_IDS_81:-${GPU_ID}}"
GPU_IDS_641="${GPU_IDS_641:-${GPU_ID}}"

SPLIT_JSON="${SPLIT_JSON:-OmniRoam/configs/train_test_files.json}"
SPLIT_SUBSET="${SPLIT_SUBSET:-test}"
IMAGE_BASE_DIR="${IMAGE_BASE_DIR:-/data1/songcx/dataset/interiogs_render}"
INTERIORGS_ROOT="${INTERIORGS_ROOT:-${IMAGE_BASE_DIR}}"

OUTPUT_ROOT_81="${OUTPUT_ROOT_81:-carla_benchmark_results/vipe_pano_pose_gt_81}"
OUTPUT_ROOT_641="${OUTPUT_ROOT_641:-carla_benchmark_results/vipe_pano_pose_gt_641}"

RESOLUTION="${RESOLUTION:-1024}"
FRAMES_SUBDIR="${FRAMES_SUBDIR:-pano_camera0}"
FRAME_EXT="${FRAME_EXT:-png}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
LOG_TO_CONSOLE="${LOG_TO_CONSOLE:-true}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

json_name="$(basename "${SPLIT_JSON}")"
json_stem="${json_name%.json}"

validate_poses() {
    local pose_root="$1"
    local frame_count="$2"

    python - "${SPLIT_JSON}" "${SPLIT_SUBSET}" "${pose_root}" "${frame_count}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

split_path = Path(sys.argv[1])
split_subset = sys.argv[2]
pose_root = Path(sys.argv[3])
frame_count = int(sys.argv[4])

split = json.loads(split_path.read_text(encoding="utf-8"))
scene_ids = split.get(split_subset)
if not isinstance(scene_ids, list):
    raise ValueError(f"{split_path} must contain split[{split_subset!r}] as a list")

missing = []
invalid = []
for scene_id in scene_ids:
    pose_path = pose_root / f"{scene_id}_poses.json"
    if not pose_path.is_file():
        missing.append(scene_id)
        continue
    try:
        poses = np.asarray(json.loads(pose_path.read_text(encoding="utf-8")))
    except Exception as error:
        invalid.append((scene_id, f"cannot read JSON: {error}"))
        continue
    expected_shape = (frame_count, 4, 4)
    if poses.shape != expected_shape:
        invalid.append((scene_id, f"shape={poses.shape}, expected={expected_shape}"))
    elif not np.isfinite(poses).all():
        invalid.append((scene_id, "contains NaN or Inf"))

print(
    f"[CHECK {frame_count}] expected={len(scene_ids)} "
    f"valid={len(scene_ids) - len(missing) - len(invalid)} "
    f"missing={len(missing)} invalid={len(invalid)}"
)
for scene_id in missing:
    print(f"[MISSING {frame_count}] {scene_id}", file=sys.stderr)
for scene_id, error in invalid:
    print(f"[INVALID {frame_count}] {scene_id}: {error}", file=sys.stderr)

if missing or invalid:
    raise SystemExit(1)
PY
}

evaluate_poses() {
    local pose_root="$1"
    local frame_count="$2"
    local pose_path pose_name video_id

    shopt -s nullglob
    local pose_paths=("${pose_root}"/*_poses.json)
    if [[ "${#pose_paths[@]}" -eq 0 ]]; then
        echo "[ERROR ${frame_count}] No pose JSON files found in ${pose_root}" >&2
        return 1
    fi

    for pose_path in "${pose_paths[@]}"; do
        pose_name="$(basename "${pose_path}")"
        video_id="${pose_name%_poses.json}"
        echo "[EVAL ${frame_count}] ${video_id}"
        VIDEO_ID="${video_id}" \
        VIPE_POSE_ROOT="${pose_root}" \
        INTERIORGS_ROOT="${INTERIORGS_ROOT}" \
        OUTPUT_ROOT="${pose_root}" \
        FRAMES_SUBDIR="${FRAMES_SUBDIR}" \
        FRAME_EXT="${FRAME_EXT}" \
        FRAME_START=1 \
        FRAME_COUNT="${frame_count}" \
        ALIGN_MODE=sim3 \
        ROTATION_MODE=absolute \
        GT_POSITION_SOURCE=location \
        GT_ROTATION_SOURCE=interiorgs_erp \
        WORLDSCORE_MODE=relative \
        PREVIEW_ROWS=0 \
        bash run_compare_vipe_omniroam_pose.sh
    done
}

run_scope() {
    local frame_count="$1"
    local gpu_ids="$2"
    local output_root="$3"
    local pose_root="${output_root}/${json_stem}_omniroam_png"

    echo "[START ${frame_count}] GPUs=${gpu_ids} output=${output_root}"
    DATASET_FORMAT=omniroam_png \
    JSON_PATH="${SPLIT_JSON}" \
    IMAGE_BASE_DIR="${IMAGE_BASE_DIR}" \
    OUTPUT_ROOT_DIR="${output_root}" \
    RESOLUTION="${RESOLUTION}" \
    SPLIT_SUBSET="${SPLIT_SUBSET}" \
    INTERIORGS_FRAMES_SUBDIR="${FRAMES_SUBDIR}" \
    INTERIORGS_MAX_FRAMES="${frame_count}" \
    INTERIORGS_FRAME_EXT="${FRAME_EXT}" \
    GPU_IDS="${gpu_ids}" \
    MULTI_PROCESS_LAUNCH=1 \
    START_CLIP_IDX=0 \
    NUM_CLIPS_PER_GPU=0 \
    LOG_TO_CONSOLE="${LOG_TO_CONSOLE}" \
    bash run_vipe_pano_pose.sh

    validate_poses "${pose_root}" "${frame_count}"

    if [[ "${RUN_EVALUATION}" == "1" || "${RUN_EVALUATION}" == "true" ]]; then
        evaluate_poses "${pose_root}" "${frame_count}"
    fi
    echo "[DONE ${frame_count}] output=${pose_root}"
}

echo "[INFO] Launching 81-frame and 641-frame GT jobs concurrently."
echo "[INFO] 81-frame GPUs=${GPU_IDS_81}; 641-frame GPUs=${GPU_IDS_641}"

run_scope 81 "${GPU_IDS_81}" "${OUTPUT_ROOT_81}" &
pid_81=$!
run_scope 641 "${GPU_IDS_641}" "${OUTPUT_ROOT_641}" &
pid_641=$!

cleanup() {
    kill -TERM "${pid_81}" "${pid_641}" 2>/dev/null || true
}
trap cleanup INT TERM

status=0
wait "${pid_81}" || status=1
wait "${pid_641}" || status=1

if [[ "${status}" -ne 0 ]]; then
    echo "[ERROR] At least one GT scope failed; inspect its output logs and rerun the same command." >&2
    exit "${status}"
fi

echo "[DONE] Both GT scopes completed successfully."
