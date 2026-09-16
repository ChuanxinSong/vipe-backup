#!/bin/bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"

INPUT_ROOT="${INPUT_ROOT:-"/home/songcx/code/omniroam/omniroam_results/zeroshot_testset_infer"}"
SCOPE="${SCOPE:-all_segments}" # first_segment | all_segments
SPLIT_JSON="${SPLIT_JSON:-omniroam/configs/train_test_files.json}"

CUDA_RESERVE_GIB="${CUDA_RESERVE_GIB:-45}"
CUDA_RESERVE_SAFETY_GIB="${CUDA_RESERVE_SAFETY_GIB:-2}"

SPLIT_SUBSET="${SPLIT_SUBSET:-test}"
EXPECTED_SEGMENTS="${EXPECTED_SEGMENTS:-8}"
VIRTUAL_VIEW_HEIGHT="${VIRTUAL_VIEW_HEIGHT:-256}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${INPUT_ROOT}/vipe_pose_eval/${SCOPE}}"

MULTI_PROCESS_LAUNCH="${MULTI_PROCESS_LAUNCH:-1}"
START_CLIP_IDX="${START_CLIP_IDX:-0}"
NUM_CLIPS_PER_GPU="${NUM_CLIPS_PER_GPU:-0}"
LOG_TO_CONSOLE="${LOG_TO_CONSOLE:-true}"

if [[ -z "${INPUT_ROOT}" ]]; then
    echo "[ERROR] INPUT_ROOT is required." >&2
    exit 2
fi
if [[ ! -d "${INPUT_ROOT}" ]]; then
    echo "[ERROR] INPUT_ROOT is not a directory: ${INPUT_ROOT}" >&2
    exit 2
fi
if [[ "${SCOPE}" != "first_segment" && "${SCOPE}" != "all_segments" ]]; then
    echo "[ERROR] SCOPE must be first_segment or all_segments, got: ${SCOPE}" >&2
    exit 2
fi

ARGS=(
    --dataset_format pose_i2v_results
    --json_path "${SPLIT_JSON}"
    --split_subset "${SPLIT_SUBSET}"
    --pose_i2v_results_root "${INPUT_ROOT}"
    --pose_i2v_scope "${SCOPE}"
    --pose_i2v_expected_segments "${EXPECTED_SEGMENTS}"
    --virtual_view_height "${VIRTUAL_VIEW_HEIGHT}"
    --output_root_dir "${OUTPUT_ROOT}"
    --cuda_reserve_gib "${CUDA_RESERVE_GIB}"
    --cuda_reserve_safety_gib "${CUDA_RESERVE_SAFETY_GIB}"
)

IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_LIST[@]}"
if [[ "${NUM_GPUS}" -le 0 ]]; then
    echo "[ERROR] GPU_IDS must contain at least one GPU id." >&2
    exit 2
fi

TOTAL_SELECTED=$(python - "${SPLIT_JSON}" "${SPLIT_SUBSET}" "${START_CLIP_IDX}" <<'PY'
import json
import sys

json_path, subset = sys.argv[1:3]
start = int(sys.argv[3])
with open(json_path, "r", encoding="utf-8") as handle:
    split = json.load(handle)
scenes = split.get(subset)
if not isinstance(scenes, list):
    raise ValueError(f"{json_path} must contain split[{subset!r}] as a list")
print(max(0, len(scenes) - max(0, start)))
PY
)
if [[ "${TOTAL_SELECTED}" -le 0 ]]; then
    echo "[INFO] No scenes selected."
    exit 0
fi

mkdir -p "${OUTPUT_ROOT}/logs"
echo "[INFO] input=${INPUT_ROOT} scope=${SCOPE} split=${SPLIT_SUBSET} virtual_view_height=${VIRTUAL_VIEW_HEIGHT}"
echo "[INFO] output=${OUTPUT_ROOT} GPUs=${GPU_IDS} start_clip_idx=${START_CLIP_IDX}"
echo "[INFO] cuda_reserve_gib=${CUDA_RESERVE_GIB} cuda_reserve_safety_gib=${CUDA_RESERVE_SAFETY_GIB} (per GPU worker)"

if [[ "${MULTI_PROCESS_LAUNCH}" == "1" || "${MULTI_PROCESS_LAUNCH}" == "true" ]]; then
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
        log_path="${OUTPUT_ROOT}/logs/run_gpu${gpu_id}_start${proc_start}.log"
        run_args=(
            "CUDA_VISIBLE_DEVICES=${gpu_id}" python -u infer_vipe_panorama_pose.py
            "${ARGS[@]}" --start_clip_idx "${proc_start}" --num_clips "${CLIPS_PER_GPU}" "$@"
        )
        printf -v run_cmd "%q " "${run_args[@]}"
        printf -v log_path_q "%q" "${log_path}"
        echo "[INFO] Launch GPU ${gpu_id}: start=${proc_start} count=${CLIPS_PER_GPU} log=${log_path}"
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
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -u infer_vipe_panorama_pose.py \
    "${ARGS[@]}" --start_clip_idx "${START_CLIP_IDX}" --num_clips "${NUM_CLIPS_PER_GPU}" "$@"
