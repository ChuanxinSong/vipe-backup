#!/bin/bash
set -euo pipefail

DATASET_FORMAT="${DATASET_FORMAT:-omniroam_png}"  # pvdepth_json | omniroam_png | omniroam_h5
JSON_PATH="${JSON_PATH:-OmniRoam/configs/train_test_files.json}"
IMAGE_BASE_DIR="${IMAGE_BASE_DIR:-/data1/songcx/dataset/interiogs_render}"
H5_DATA_ROOT="${H5_DATA_ROOT:-}"
OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-carla_benchmark_results/vipe_pano_pose}"
RESOLUTION="${RESOLUTION:-1024}"
SPLIT_SUBSET="${SPLIT_SUBSET:-test}"
INTERIORGS_FRAMES_SUBDIR="${INTERIORGS_FRAMES_SUBDIR:-pano_camera0}"
INTERIORGS_MAX_FRAMES="${INTERIORGS_MAX_FRAMES:-800}"
INTERIORGS_FRAME_EXT="${INTERIORGS_FRAME_EXT:-png}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"
MULTI_PROCESS_LAUNCH="${MULTI_PROCESS_LAUNCH:-1}"
START_CLIP_IDX="${START_CLIP_IDX:-0}"
NUM_CLIPS_PER_GPU="${NUM_CLIPS_PER_GPU:-0}"
LOG_TO_CONSOLE="${LOG_TO_CONSOLE:-true}"

ARGS=(
    --dataset_format "${DATASET_FORMAT}"
    --json_path "${JSON_PATH}"
    --output_root_dir "${OUTPUT_ROOT_DIR}"
    --resolution "${RESOLUTION}"
)

if [[ "${DATASET_FORMAT}" == "pvdepth_json" || "${DATASET_FORMAT}" == "omniroam_png" ]]; then
    ARGS+=(--image_base_dir "${IMAGE_BASE_DIR}")
fi

if [[ "${DATASET_FORMAT}" == "omniroam_png" || "${DATASET_FORMAT}" == "omniroam_h5" ]]; then
    ARGS+=(
        --split_subset "${SPLIT_SUBSET}"
        --interiorgs_frames_subdir "${INTERIORGS_FRAMES_SUBDIR}"
        --interiorgs_max_frames "${INTERIORGS_MAX_FRAMES}"
        --interiorgs_frame_ext "${INTERIORGS_FRAME_EXT}"
    )
fi

if [[ "${DATASET_FORMAT}" == "omniroam_h5" ]]; then
    ARGS+=(--h5_data_root "${H5_DATA_ROOT}")
fi

# Pass --overwrite to this script to re-estimate clips with existing pose files.
IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_LIST[@]}"
if [[ "${NUM_GPUS}" -le 0 ]]; then
    echo "[ERROR] GPU_IDS must contain at least one GPU id." >&2
    exit 1
fi

echo "[INFO] GPUs=${GPU_IDS} dataset=${DATASET_FORMAT} split=${SPLIT_SUBSET} resolution=${RESOLUTION}"
echo "[INFO] json=${JSON_PATH} output=${OUTPUT_ROOT_DIR} start_clip_idx=${START_CLIP_IDX}"

if [[ "${MULTI_PROCESS_LAUNCH}" == "1" || "${MULTI_PROCESS_LAUNCH}" == "true" ]]; then
    TOTAL_SELECTED=$(python - "${DATASET_FORMAT}" "${JSON_PATH}" "${SPLIT_SUBSET}" "${START_CLIP_IDX}" <<'PY'
import json
import sys

dataset_format, json_path, split_subset = sys.argv[1:4]
start_clip_idx = int(sys.argv[4])

with open(json_path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

if dataset_format == "pvdepth_json":
    if not isinstance(data, dict):
        raise ValueError(f"{json_path} must contain a JSON object at the top level.")
    total = 0
    for town_data in data.values():
        if not isinstance(town_data, dict):
            continue
        for path_data in town_data.values():
            if not isinstance(path_data, dict):
                continue
            total += sum(1 for frames in path_data.values() if isinstance(frames, list))
else:
    scenes = data.get(split_subset)
    if not isinstance(scenes, list):
        raise ValueError(f"{json_path} must contain split[{split_subset!r}] as a list.")
    total = len(scenes)

print(max(0, total - max(0, start_clip_idx)))
PY
)
    if [[ "${TOTAL_SELECTED}" -le 0 ]]; then
        echo "[INFO] No clips selected"
        exit 0
    fi
    mkdir -p "${OUTPUT_ROOT_DIR}/logs"
    if [[ "${NUM_CLIPS_PER_GPU}" -gt 0 ]]; then
        CLIPS_PER_GPU="${NUM_CLIPS_PER_GPU}"
    else
        CLIPS_PER_GPU=$(( (TOTAL_SELECTED + NUM_GPUS - 1) / NUM_GPUS ))
    fi
    echo "[INFO] total_selected=${TOTAL_SELECTED} clips_per_gpu=${CLIPS_PER_GPU}"

    PIDS=()
    cleanup() {
        for pid in "${PIDS[@]}"; do
            kill -TERM -- "-${pid}" 2>/dev/null || true
        done
        exit 130
    }
    trap cleanup INT TERM

    for gpu_index in "${!GPU_LIST[@]}"; do
        gpu_id="${GPU_LIST[$gpu_index]//[[:space:]]/}"
        proc_start=$(( START_CLIP_IDX + gpu_index * CLIPS_PER_GPU ))
        if [[ $((proc_start - START_CLIP_IDX)) -ge "${TOTAL_SELECTED}" ]]; then
            continue
        fi
        log_path="${OUTPUT_ROOT_DIR}/logs/run_gpu${gpu_id}_start${proc_start}.log"
        run_args=(
            "CUDA_VISIBLE_DEVICES=${gpu_id}" python -u infer_vipe_panorama_pose.py
            "${ARGS[@]}" --start_clip_idx "${proc_start}" --num_clips "${CLIPS_PER_GPU}" "$@"
        )
        printf -v run_cmd "%q " "${run_args[@]}"
        printf -v log_path_q "%q" "${log_path}"
        echo "[INFO] Launch GPU ${gpu_id}: start_clip_idx=${proc_start} num_clips=${CLIPS_PER_GPU} log=${log_path}"
        if [[ "${LOG_TO_CONSOLE}" == "1" || "${LOG_TO_CONSOLE}" == "true" ]]; then
            setsid bash -c "${run_cmd} 2>&1 | tee ${log_path_q}" &
        else
            setsid bash -c "${run_cmd} > ${log_path_q} 2>&1" &
        fi
        PIDS+=("$!")
    done

    failed=0
    for pid in "${PIDS[@]}"; do
        wait "${pid}" || failed=1
    done
    exit "${failed}"
else
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -u infer_vipe_panorama_pose.py \
        "${ARGS[@]}" --start_clip_idx "${START_CLIP_IDX}" --num_clips "${NUM_CLIPS_PER_GPU}" "$@"
fi
