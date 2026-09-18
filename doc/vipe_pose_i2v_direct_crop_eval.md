# ViPE 直接读取 Pose-I2V Comparison PNG 并评估位姿

本文说明如何把 OmniRoam Pose-I2V 生成结果直接送入 ViPE：读取每个 comparison PNG，依据 metadata 在内存中裁出上半部分的预测全景图，估计相机位姿，再与 InteriorGS `transforms.json` GT 对齐并计算指标。

整个流程不会提前导出或保存裁剪图。

## 1. 环境准备

完整的新机器安装过程见 [vipe_panorama_pose_new_machine_setup.md](vipe_panorama_pose_new_machine_setup.md)。本流程使用同一个环境：

```bash
cd /path/to/vipe
conda activate vipe-panorama
```

先检查 Python、PyTorch、GPU 和 ViPE 扩展：

```bash
python --version
python -c "
import torch
import vipe_ext
print('torch:', torch.__version__)
print('torch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
print('vipe_ext:', vipe_ext.__file__)
"
```

运行已经编译好的 ViPE 时不要求 shell 中能找到 `nvcc`；只有首次安装或重新编译 `vipe_ext` 时才需要 CUDA 编译器。如果扩展还没有编译且 `nvcc` 不存在，按照新机器部署手册的“安装或修复 nvcc”和“编译 ViPE”章节处理。

指标阶段显式使用 SciPy 求 WorldScore scalar scale，确认环境中可以导入：

```bash
python -c "import scipy; print(scipy.__version__)"
```

所有命令都应从 ViPE 仓库根目录运行。

## 2. 输入目录和 metadata 契约

`INPUT_ROOT` 指向一次 Pose-I2V 实验的结果根目录，例如：

```text
OmniRoam/omniroam_results/
└── pose_render_i2v_timestep_channel_gate_testset_learnedChannelGate_scalar_mlp/
    ├── 0007_840137/
    │   ├── metadata.json
    │   ├── segment_00/frames/frame_0001.png
    │   ├── ...
    │   └── segment_07/frames/frame_0641.png
    └── ...
```

OmniRoam comparison 格式的场景 `metadata.json` 必须包含：

- 与目录名一致的 `scene_id`；
- 非空 `segments`；
- 从 0 开始连续的 `segment_id`；
- 每段非空的 `saved_frame_indices`；
- 每段 `frame_layout.generated_crop=[left, top, right, bottom]`。

crop 坐标使用半开区间。对于当前 `640×642` comparison PNG：

```json
{
  "generated_crop": [0, 0, 640, 320]
}
```

等价于：

```python
generated = image[0:320, 0:640]
```

读取器会检查 crop 未越界、尺寸在所有帧中一致，并且裁出的全景图宽高比严格为 2:1。它按 `saved_frame_indices` 取帧，因此后续 segment 中未保存的边界重复帧不会被重复送入 ViPE。

metadata 采用结构校验，不限制 `output_format_version` 的具体字符串；版本仍会记录到输出 manifest 中。

### PanoWorld raw 兼容格式

读取器也支持 `format_version=panoworld_interiogs_rollout_v1`。该格式仍然保存上下拼接 comparison PNG，但 metadata 和文件名不同：

```text
<INPUT_ROOT>/<scene_id>/
├── metadata.json
├── segment_00/frames/000001.png
├── ...
└── segment_07/frames/000641.png
```

PanoWorld 每个 segment 使用 `saved_frame_ids`，图片使用六位数文件名。metadata 不提供 `frame_layout`，读取器只对明确的 `panoworld_interiogs_rollout_v1` 格式使用 segment 的 `width` 和 `height` 推导布局：

```text
comparison PNG：width × (2*height+2)
预测 ERP crop：[0, 0, width, height]
分隔条：       [0, height, width, height+2]
GT ERP：       [0, height+2, width, 2*height+2]
```

当前 PanoWorld 数据对应 `960×962 -> 960×480`。读取器会严格检查 `width=2*height`、comparison PNG 尺寸、六位数文件是否存在、frame ID 全局递增以及所有 crop 尺寸一致；未知的无 `frame_layout` 格式仍会拒绝，避免猜测 crop 后误用 GT 图像。

## 3. 两种评估范围

### first_segment

只读取 `segment_00`。其他 7 段可以不存在或尚未完成。当前数据通常对应：

```text
frame 1..81，共 81 帧
```

### all_segments

默认严格要求 8 段，即 `segment_00..segment_07`。OmniRoam 按每段 `saved_frame_indices`、PanoWorld 按 `saved_frame_ids` 顺序拼接，并要求全局 frame ID 严格递增、不重复。当前数据通常对应：

```text
frame 1..641，共 641 帧
```

两个范围需要分别运行推理和评估，结果放在独立目录中。

## 4. 实际图像尺寸链路

Pose-I2V 输入不会 resize 到 `1024×512`：

```text
comparison PNG 640×642
  -> metadata crop，得到原生预测 ERP 640×320
  -> EquirectProjectionProcessor，得到 5 个 256×256 pinhole 视图
  -> StandardResizeStreamProcessor，缩放到 443×443
  -> 中心裁剪到 440×440
  -> ViPE SLAM
```

PanoWorld 使用同一条内存 crop 和投影链路，只是原生 comparison/crop 尺寸不同：

```text
comparison PNG 960×962
  -> format_version + width/height 推导 crop，得到预测 ERP 960×480
  -> EquirectProjectionProcessor，得到 5 个 256×256 pinhole 视图
  -> StandardResizeStreamProcessor，缩放并中心裁剪到 440×440
  -> ViPE SLAM
```

5 个视图来自 `configs/pipeline/panorama.yaml`：4 个水平视角和 1 个 bottom 视角，FOV 为 100 度。

固定 `256×256` 是为了保持原有 `--resolution=1024` 基线的虚拟相机内参和中间投影视图尺寸。SLAM 自己按固定目标面积缩放输入，所以 `256×256` 最终仍会变成 `440×440`。先把 `640×320` ERP 放大为 `1024×512` 不会产生新信息，只会增加一次插值，因此新输入路径直接从原生 ERP 投影。

原有 `pvdepth_json`、`omniroam_png` 和 `omniroam_h5` 不受影响：没有显式传 `--virtual_view_height` 时，它们仍使用 `resolution/4` 的虚拟视图高度，并保留原来的 ERP resize 行为。

## 5. 运行 ViPE 位姿估计

先设置输入：

```bash
export INPUT_ROOT="OmniRoam/omniroam_results/pose_render_i2v_timestep_channel_gate_testset_learnedChannelGate_scalar_mlp"
export SPLIT_JSON="OmniRoam/configs/train_test_files.json"
```

### 单 GPU：首 segment

```bash
conda activate vipe-panorama

INPUT_ROOT="${INPUT_ROOT}" \
SPLIT_JSON="${SPLIT_JSON}" \
SCOPE=first_segment \
GPU_IDS=0 \
bash run_vipe_pose_i2v.sh
```

### 多 GPU：完整 8 segments

```bash
INPUT_ROOT="${INPUT_ROOT}" \
SPLIT_JSON="${SPLIT_JSON}" \
SCOPE=all_segments \
EXPECTED_SEGMENTS=8 \
VIRTUAL_VIEW_HEIGHT=256 \
GPU_IDS=0,1,2,3 \
bash run_vipe_pose_i2v.sh
```

脚本按 split 中的场景顺序连续分片给各 GPU。常用控制参数：

```text
SPLIT_SUBSET=test
START_CLIP_IDX=0
NUM_CLIPS_PER_GPU=0       # 0 表示自动均分
MULTI_PROCESS_LAUNCH=1
LOG_TO_CONSOLE=true
OUTPUT_ROOT=<INPUT_ROOT>/vipe_pose_eval/<scope>
CUDA_RESERVE_GIB=0             # 0 关闭；大于 0 时每个 GPU worker 提前缓存到该总显存量
CUDA_RESERVE_SAFETY_GIB=2      # 预留完成后至少保留的全局空闲显存
```

### 5.1 共享 GPU 上提前预留显存

ViPE 的图像 buffer 会较早分配，但前端 factor graph 的 correlation volume 和优化状态会随关键帧逐步增加，因此 `nvidia-smi` 中的显存占用通常不会在启动时立即达到峰值。如果共享机器没有调度器或独占 GPU，可以让每个 ViPE worker 在推理前把 PyTorch CUDA cache 提前扩展到预计峰值：

```bash
INPUT_ROOT="${INPUT_ROOT}" \
SCOPE=all_segments \
GPU_IDS=0,1 \
CUDA_RESERVE_GIB=22 \
CUDA_RESERVE_SAFETY_GIB=2 \
bash run_vipe_pose_i2v.sh
```

`CUDA_RESERVE_GIB` 表示该 worker 的**目标总 reserved 显存**，不是在已有占用之外再增加同样大小。例如模型已 reserved 2 GiB、目标为 22 GiB 时，只会补足约 20 GiB。脚本先创建临时 CUDA tensor，再删除 Python tensor；PyTorch caching allocator 会保留底层分配供后续 SLAM 张量复用。日志会输出预留前后的 `reserved`、`allocated` 和全局空闲量。

使用建议：

- 默认值为 `0`，现有运行行为不变；
- 先在空闲 GPU 上完整运行一个代表性的 `all_segments` 场景，按 `torch.cuda.max_memory_reserved()` 峰值再加 1--2 GiB 设置目标；
- `CUDA_RESERVE_SAFETY_GIB` 是留给 CUDA context、算子 workspace 和非 PyTorch CUDA 分配的安全余量，不建议设为 0；
- 每个多 GPU worker 只预留其 `CUDA_VISIBLE_DEVICES` 映射后的逻辑 GPU 0，不会在一张卡上重复预留；
- 显式请求无法满足时进程会报出目标、现有 reserved、全局空闲量并退出，不会静默降级；
- 不要在推理期间调用 `torch.cuda.empty_cache()`，否则空闲 cache 会归还给 CUDA；
- `torch.cuda.set_per_process_memory_fraction()` 只是限制上限，不能达到提前占用效果；
- 这是一种无调度共享机器上的工程性保护，能够使用 Slurm 等系统时仍应优先申请独占 GPU。

只调试一个场景时：

```bash
INPUT_ROOT="${INPUT_ROOT}" \
SCOPE=first_segment \
GPU_IDS=0 \
START_CLIP_IDX=0 \
NUM_CLIPS_PER_GPU=1 \
bash run_vipe_pose_i2v.sh
```

已有 pose 和匹配的有效 manifest 会被跳过。重新计算时追加 `--overwrite`：

```bash
INPUT_ROOT="${INPUT_ROOT}" SCOPE=first_segment GPU_IDS=0 \
bash run_vipe_pose_i2v.sh --overwrite
```

单场景失败不会阻止后续场景运行，但只要存在失败，脚本最终返回非零状态。详细异常位于 `<OUTPUT_ROOT>/logs/`。

## 6. Pose 与 input manifest

每个场景产生：

```text
<OUTPUT_ROOT>/poses/<scene_id>_poses.json
<OUTPUT_ROOT>/poses/<scene_id>_input_manifest.json
```

pose JSON 形状为 `[T,4,4]`，内容是 ViPE 输出的逐帧 C2W。

manifest 记录：

- scene、scope、metadata 路径和源格式版本；
- 每个 pose 对应的真实 `frame_id`、`segment_id` 和 comparison PNG；
- 公共 `generated_crop`、逐 segment crop 映射和原生 crop 尺寸；不同 segment 可以使用不同坐标，但裁出尺寸必须一致；
- 虚拟视图尺寸和 SLAM 实际输入尺寸；
- 虚拟相机 FOV、视图数和 top/bottom 配置。

评估必须使用这个 manifest 中的 frame ID，而不能假设 pose 下标总是等于连续 GT frame ID。

## 7. 与 GT 对齐并计算指标

设置 InteriorGS 根目录，其中每个场景应有 `<scene_id>/transforms.json`：

```bash
export INTERIORGS_ROOT=/path/to/interiogs_render
```

评估首 segment：

```bash
conda activate vipe-panorama

INPUT_ROOT="${INPUT_ROOT}" \
SCOPE=first_segment \
SPLIT_JSON="${SPLIT_JSON}" \
INTERIORGS_ROOT="${INTERIORGS_ROOT}" \
bash run_compare_vipe_pose_i2v.sh
```

评估完整 8 segments：

```bash
INPUT_ROOT="${INPUT_ROOT}" \
SCOPE=all_segments \
SPLIT_JSON="${SPLIT_JSON}" \
INTERIORGS_ROOT="${INTERIORGS_ROOT}" \
bash run_compare_vipe_pose_i2v.sh
```

默认评估配置为：

```text
ALIGN_MODE=sim3
ROTATION_MODE=constant_offset
GT_POSITION_SOURCE=location
GT_ROTATION_SOURCE=interiorgs_erp
WORLDSCORE_MODE=relative
WORLDSCORE_SCALE_SOLVER=scipy
```

`scipy` 是显式固定的；SciPy 缺失或求解失败时会直接报错，不会因另一台机器安装了 cvxpy 而切换算法。

## 8. 输出和指标解释

每个 scope 的完整目录为：

```text
<INPUT_ROOT>/vipe_pose_eval/<scope>/
├── poses/
│   ├── <scene_id>_poses.json
│   └── <scene_id>_input_manifest.json
├── metrics/
│   ├── <scene_id>_vipe_vs_gt.csv
│   └── <scene_id>_vipe_vs_gt_summary.json
├── metrics_per_scene.csv
├── metrics_summary.json
└── logs/
```

主指标是：

```text
worldscore_rotation_mean_deg   越低越好
worldscore_translation_mean    越低越好
```

`metrics_summary.json` 对成功场景的这两个值计算场景级 macro average，每个场景权重相同，不会因为 `all_segments` 的帧数更长而增加权重。

单场景 summary 和 CSV 还保留：

- WorldScore scale、逐帧旋转和平移误差；
- Sim3 对齐后的平移误差；
- constant rotation offset 后的旋转误差；
- 相邻帧 step translation/rotation diagnostics。

`translation_rmse_after` 是 Sim3 diagnostic，不是主 WorldScore translation 指标。对外汇报时优先使用 `worldscore_rotation_mean_deg` 和 `worldscore_translation_mean`，并明确 scope 是 `first_segment` 还是 `all_segments`。

批量评估遇到缺 pose、manifest、GT 或单场景计算失败时，会把原因写入 `metrics_per_scene.csv` 和 `metrics_summary.json`，继续其他场景，最后返回非零状态。

## 9. 单场景手动评估

需要排查某个场景时，可以直接调用原有单场景比较器的新接口：

```bash
python compare_vipe_omniroam_pose.py \
  --vipe_poses "${INPUT_ROOT}/vipe_pose_eval/first_segment/poses/0007_840137_poses.json" \
  --frame_manifest "${INPUT_ROOT}/vipe_pose_eval/first_segment/poses/0007_840137_input_manifest.json" \
  --gt_transforms_json "${INTERIORGS_ROOT}/0007_840137/transforms.json" \
  --worldscore_mode relative \
  --worldscore_scale_solver scipy \
  --preview_rows 10
```

不传 `--frame_manifest` 时，比较器仍支持旧的 `--frame_start` 和 `--frame_count` 连续帧模式，因此原有 `run_compare_vipe_omniroam_pose.sh` 保持兼容。

## 10. 快速自检

运行不需要模型推理的测试：

```bash
conda activate vipe-panorama
python -m unittest discover -s tests -p 'test_pose_i2v_pipeline.py' -v
```

测试覆盖 metadata crop、scope 拼接、重复帧、越界 crop、`256→440` 尺寸链路、非连续 GT frame ID 和 SciPy scale solver。

正式跑全量前，建议先完成一个 `first_segment` 场景，确认：

```text
pose shape              = [81,4,4]
manifest frame_count    = 81
crop_size_hw            = [320,640]
virtual_view_size_hw    = [256,256]
slam_input_size_hw      = [440,440]
worldscore_scale_solver = scipy_minimize_scalar
```
