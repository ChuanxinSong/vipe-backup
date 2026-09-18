# ViPE 全景位姿与 OmniRoam 指标：新机器部署手册（CUDA 12.4）

本文用于在一台新机器上从零完成以下流程：

```text
安装系统和 Conda 环境
  -> 编译 ViPE CUDA 扩展
  -> 准备 DROID-SLAM 权重
  -> 准备 OmniRoam / InteriorGS 数据
  -> ViPE 估计全景序列位姿
  -> 与 transforms.json GT 对比
  -> 汇总 WorldScore-style 指标
```

本文专门面向不能使用 `torch 2.7.0+cu128`、但能够运行 CUDA 12.4 的机器，使用以下稳定组合：

```text
PyTorch       2.6.0+cu124
Torchvision   0.21.0+cu124
CUDA / nvcc   12.4
Python        3.10
```

不要尝试安装 `torch 2.7.0+cu124`：PyTorch 没有发布这个稳定版组合。本教程也不适用于必须依赖 CUDA 12.8 或更高版本的 Blackwell GPU。

版本依据：

- [PyTorch Previous Versions](https://docs.pytorch.org/get-started/previous-versions/) 给出的 CUDA 12.4 稳定组合是 `torch==2.6.0` 与 `torchvision==0.21.0`。
- [NVIDIA CUDA 12.4 Release Notes](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/) 给出了 CUDA 12.4 对应驱动及 CUDA 12.x minor-version compatibility 的最低驱动要求。

相对于 cu128 环境，本教程只更换 Torch、Torchvision、Triton 和 Torch 自带的 NVIDIA CUDA 运行库。ViPE 配置、DROID 权重、输入数据、推理命令和评测定义保持不变。

本文对应的主要入口是：

```text
run_vipe_pano_pose.sh
infer_vipe_panorama_pose.py
run_compare_vipe_omniroam_pose.sh
compare_vipe_omniroam_pose.py
```

评测定义和坐标系的详细解释见：

```text
doc/vipe_panorama_pose_and_omniroam_eval.md
```

## 1. 固定代码版本

为了让不同机器尽量使用相同代码，首先固定一个同时包含本教程和以下文件的 Git commit：

```text
envs/base-cu124.yml
envs/requirements-cu124.in
envs/requirements-cu124.txt
```

这些内容位于 `vipe-backup` 的 `dev_panorama` 分支，不在当前上游 `nv-tlabs/vipe` 的远端分支中。在准备代码的机器上先记录 commit：

```bash
git rev-parse HEAD
```

在新机器上克隆仓库。如果需要严格复现，请把下面的 `<CU124_COMMIT>` 替换成上一步记录的 commit：

```bash
git clone \
  --branch dev_panorama \
  https://github.com/ChuanxinSong/vipe-backup.git \
  /path/to/vipe

cd /path/to/vipe
git checkout <CU124_COMMIT>

test -f envs/base-cu124.yml
test -f envs/requirements-cu124.txt
```

原 cu128 教程中记录的 `535b55113af6a2922253ef8932245eefe6f0738a` 是算法代码基线，但它本身早于本教程及 cu124 环境文件；不要在 checkout 该旧 commit 后直接寻找 `envs/*-cu124.*`。

如果该仓库需要权限，请先在新机器配置对应的 GitHub 凭据或改用有权限的 SSH URL。

后续所有 `bash run_*.sh` 命令都应该从 ViPE 仓库根目录运行，因为 shell 脚本使用了相对路径调用 Python 文件。

建议先定义几个只用于本次任务的路径变量。请根据新机器实际路径修改：

```bash
export VIPE_REPO=/path/to/vipe
export OMNIROAM_SPLIT_JSON=/data/omniroam/train_test_files.json
export INTERIORGS_DATA_ROOT=/data/interiogs_render
export VIPE_RESULT_ROOT=/data/results/vipe_pano_pose
export VIPE_EVAL_ROOT=/data/results/vipe_pano_pose_eval
export TORCH_HOME=/data/model_cache/torch

cd "${VIPE_REPO}"
```

## 2. 机器和驱动要求

推荐配置：

```text
操作系统：Linux x86_64
Python：3.10
PyTorch：2.6.0+cu124
CUDA 编译工具链：12.4
GPU：NVIDIA GPU
显存：1024 分辨率、800 帧时建议至少约 45 GB 空闲显存，80 GB 更稳
```

检查 GPU 和驱动：

```bash
nvidia-smi
```

建议 Linux 驱动版本不低于 `550.54.14`，这样与 CUDA 12.4 原生匹配。CUDA 12.x 的 minor-version compatibility 最低驱动是 `525.60.13`，但低于原生匹配版本时可能受到功能限制；低于 `525.60.13` 时不要继续安装 cu124，应升级驱动或改用更低 CUDA 版本。

注意，`nvidia-smi` 顶部显示的 `CUDA Version` 表示驱动最高支持的 CUDA 版本，不代表当前 Conda 环境已经安装了对应 toolkit。真正用于本教程的运行库版本和编译器版本需要分别通过 `torch.version.cuda` 与 `nvcc --version` 确认。

这里最重要的是：

1. 系统可以正常识别 NVIDIA GPU。
2. NVIDIA 驱动能够运行 CUDA 12.4 构建的 PyTorch。
3. 正式推理时 GPU 上没有其他占用大量显存的进程。

还需要基本编译工具：

```bash
git --version
gcc --version
g++ --version
```

如果缺少 `gcc/g++`，请先使用机器对应的系统包管理器安装 C/C++ build tools。

## 3. 安装 Conda

如果机器上已经有 Conda，可以跳过本节。

安装 Miniconda 或 Anaconda 后，确认：

```bash
conda --version
```

如果当前 shell 尚未初始化 Conda，按 Conda 安装位置执行初始化，然后重新打开 shell。例如：

```bash
conda init bash
```

## 4. 创建 `vipe-panorama-cu124` 环境

仓库提供的 `envs/base-cu124.yml` 包含 Python 3.10 和 CUDA 12.4 编译依赖。创建环境：

```bash
cd "${VIPE_REPO}"

conda env create \
  -n vipe-panorama-cu124 \
  -f envs/base-cu124.yml

conda activate vipe-panorama-cu124
```

安装锁定的 Python 依赖：

```bash
python -m pip install -r envs/requirements-cu124.txt
```

不要在这个环境中安装原来的 `envs/requirements.txt`，因为该文件会重新引入 `torch 2.7.0+cu128` 及 CUDA 12.8 运行库。

锁定环境中的关键版本应为：

```text
Python        3.10.x
PyTorch       2.6.0+cu124
Torchvision   0.21.0+cu124
NumPy         2.1.2
SciPy         1.15.1
OpenCV        4.11.0
OmegaConf     2.3.0
```

检查 Python 和 PyTorch：

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('PyTorch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU 0:', torch.cuda.get_device_name(0))
"

python -m pip check
```

预期至少包含：

```text
PyTorch: 2.6.0+cu124
PyTorch CUDA build: 12.4
CUDA available: True
```

## 5. 安装或修复 `nvcc`

PyTorch 的 `cu124` wheel 包含 CUDA 运行库，但不保证包含 CUDA 编译器。ViPE 首次安装需要用 `nvcc` 编译自定义扩展。

先检查：

```bash
which nvcc
nvcc --version
```

如果出现：

```text
bash: nvcc: command not found
```

在 `vipe-panorama-cu124` 环境内安装仓库所需的 CUDA 12.4 编译组件：

```bash
conda activate vipe-panorama-cu124

conda install \
  --override-channels \
  -c nvidia/label/cuda-12.4.0 \
  -c conda-forge \
  cuda-nvcc \
  eigen \
  zlib \
  libcusparse-dev \
  libcublas-dev \
  libcusolver-dev
```

设置当前 shell 的 CUDA 编译路径：

```bash
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
hash -r
```

再次确认：

```bash
which nvcc
nvcc --version
```

预期 `nvcc` 位于当前 Conda 环境，例如：

```text
.../envs/vipe-panorama-cu124/bin/nvcc
Cuda compilation tools, release 12.4
```

不建议直接安装系统仓库中的 `nvidia-cuda-toolkit`，因为它可能提供与 `torch==2.6.0+cu124` 不一致的 CUDA 版本。

## 6. 编译并安装 ViPE

确认已经激活环境，并且当前目录是仓库根目录：

```bash
conda activate vipe-panorama-cu124
cd "${VIPE_REPO}"

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
```

编译并以 editable 方式安装：

```bash
python -m pip install \
  -v \
  --no-build-isolation \
  --no-deps \
  -e .
```

必须在 cu124 环境中重新编译。不要从 cu128 机器复制 `vipe_ext*.so`；如果工作目录曾在其他 Torch/CUDA 环境中编译过，优先使用全新的 Git checkout。

该步骤会编译类似下面的文件：

```text
vipe_ext.cpython-310-x86_64-linux-gnu.so
```

验证安装：

```bash
python -c "
import vipe
import vipe_ext
print('ViPE version:', vipe.__version__)
print('ViPE CUDA extension:', vipe_ext.__file__)
"
```

如果 `import vipe_ext` 失败，不要继续推理。优先检查：

```bash
which python
which nvcc
python -m pip show vipe
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

确认编译和运行使用的是同一个 `vipe-panorama-cu124` 环境。若仓库是从其他机器直接复制过来的，旧 `.so` 可能与新机器不兼容；最稳妥的做法是在干净的仓库 checkout 中重新执行本节的安装命令。

## 7. 准备 DROID-SLAM 权重

纯位姿流程使用 DROID-SLAM 权重：

```text
droid.pth
```

设置统一的 Torch 缓存目录：

```bash
export TORCH_HOME=/data/model_cache/torch
mkdir -p "${TORCH_HOME}/hub/droid_slam"
```

首次创建 `DroidNet` 时，代码会自动从 Google Drive 下载到：

```text
${TORCH_HOME}/hub/droid_slam/droid.pth
```

也可以在浏览器中打开下面的 Google Drive 链接手动下载：

```text
https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view
```

下载完成后，将文件重命名为 `droid.pth`，并放到代码所使用的缓存路径：

```bash
mkdir -p "${TORCH_HOME}/hub/droid_slam"
cp /path/to/downloaded/droid.pth "${TORCH_HOME}/hub/droid_slam/droid.pth"
```

其中 `/path/to/downloaded/droid.pth` 需要替换为浏览器实际下载到的文件路径。如果新机器无法访问 Google Drive，也可以从已运行成功的机器复制该文件到相同位置。

当前已验证文件的 SHA256 是：

```text
46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526
```

复制后检查：

```bash
sha256sum "${TORCH_HOME}/hub/droid_slam/droid.pth"
```

当前 pose-only 配置设置了：

```text
keyframe_depth = None
sparse_tracks.name = dummy
```

因此不需要 Unik3D、DepthCrafter、Metric3D 或 SuperPoint 权重。

## 8. 准备 OmniRoam / InteriorGS 数据

推荐使用明确的绝对路径，不依赖当前开发机器上的软链接。需要如下结构：

```text
/data/omniroam/train_test_files.json

/data/interiogs_render/
├── 0007_840137/
│   ├── pano_camera0/
│   │   ├── frame_0001.png
│   │   ├── frame_0002.png
│   │   ├── ...
│   │   └── frame_0800.png
│   └── transforms.json
├── 0026_839976/
│   └── ...
└── ...
```

文件用途：

```text
train_test_files.json             test scene ID 列表
pano_camera0/frame_XXXX.png       ViPE 位姿推理输入
transforms.json                   指标计算使用的 GT
```

当前 test split 包含：

```text
98 个 scene
每个 scene 800 帧
总计 78,400 张全景图
当前数据约 130 GiB
```

重新设置新机器路径：

```bash
export OMNIROAM_SPLIT_JSON=/data/omniroam/train_test_files.json
export INTERIORGS_DATA_ROOT=/data/interiogs_render
```

先检查一个 scene：

```bash
test -f "${OMNIROAM_SPLIT_JSON}"
test -f "${INTERIORGS_DATA_ROOT}/0007_840137/pano_camera0/frame_0001.png"
test -f "${INTERIORGS_DATA_ROOT}/0007_840137/pano_camera0/frame_0800.png"
test -f "${INTERIORGS_DATA_ROOT}/0007_840137/transforms.json"
```

可以使用下面的脚本检查 test split 的所有输入：

```bash
python - "${OMNIROAM_SPLIT_JSON}" "${INTERIORGS_DATA_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

split_path = Path(sys.argv[1])
data_root = Path(sys.argv[2])
split = json.loads(split_path.read_text(encoding="utf-8"))
video_ids = split["test"]

missing = []
for video_id in video_ids:
    scene = data_root / video_id
    transforms = scene / "transforms.json"
    if not transforms.is_file():
        missing.append(str(transforms))
    for frame_id in range(1, 801):
        frame = scene / "pano_camera0" / f"frame_{frame_id:04d}.png"
        if not frame.is_file():
            missing.append(str(frame))

print(f"test scenes: {len(video_ids)}")
print(f"missing files: {len(missing)}")
for path in missing[:20]:
    print(path)

if missing:
    raise SystemExit(1)
PY
```

必须在 `missing files: 0` 后再开始完整推理。

## 9. 运行短序列 smoke test

不要一上来跑全部 98 个 scene。先用一个 scene 的前 20 帧验证环境、权重、CUDA 扩展和数据读取。

使用独立输出目录，避免与正式结果混合：

```bash
cd "${VIPE_REPO}"
conda activate vipe-panorama-cu124

export TORCH_HOME=/data/model_cache/torch

DATASET_FORMAT=omniroam_png \
JSON_PATH="${OMNIROAM_SPLIT_JSON}" \
IMAGE_BASE_DIR="${INTERIORGS_DATA_ROOT}" \
OUTPUT_ROOT_DIR=/data/results/vipe_pano_pose_smoke \
RESOLUTION=1024 \
SPLIT_SUBSET=test \
INTERIORGS_FRAMES_SUBDIR=pano_camera0 \
INTERIORGS_MAX_FRAMES=20 \
INTERIORGS_FRAME_EXT=png \
GPU_IDS=0 \
MULTI_PROCESS_LAUNCH=0 \
START_CLIP_IDX=0 \
NUM_CLIPS_PER_GPU=1 \
LOG_TO_CONSOLE=true \
bash run_vipe_pano_pose.sh
```

如果 split 文件名是 `train_test_files.json`，预期输出为：

```text
/data/results/vipe_pano_pose_smoke/
└── train_test_files_omniroam_png/
    └── 0007_840137_poses.json
```

检查输出：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path(
    "/data/results/vipe_pano_pose_smoke/"
    "train_test_files_omniroam_png/"
    "0007_840137_poses.json"
)
poses = json.loads(path.read_text(encoding="utf-8"))
print("frames:", len(poses))
print("pose shape:", len(poses[0]), len(poses[0][0]))
PY
```

预期：

```text
frames: 20
pose shape: 4 4
```

## 10. 运行完整 ViPE 位姿推理

### 10.1 单 GPU

```bash
cd "${VIPE_REPO}"
conda activate vipe-panorama-cu124

export TORCH_HOME=/data/model_cache/torch
export VIPE_RESULT_ROOT=/data/results/vipe_pano_pose

DATASET_FORMAT=omniroam_png \
JSON_PATH="${OMNIROAM_SPLIT_JSON}" \
IMAGE_BASE_DIR="${INTERIORGS_DATA_ROOT}" \
OUTPUT_ROOT_DIR="${VIPE_RESULT_ROOT}" \
RESOLUTION=1024 \
SPLIT_SUBSET=test \
INTERIORGS_FRAMES_SUBDIR=pano_camera0 \
INTERIORGS_MAX_FRAMES=800 \
INTERIORGS_FRAME_EXT=png \
GPU_IDS=0 \
MULTI_PROCESS_LAUNCH=1 \
START_CLIP_IDX=0 \
NUM_CLIPS_PER_GPU=0 \
LOG_TO_CONSOLE=true \
bash run_vipe_pano_pose.sh
```

### 10.2 多 GPU

例如使用物理 GPU 0 和 1：

```bash
cd "${VIPE_REPO}"
conda activate vipe-panorama-cu124

export TORCH_HOME=/data/model_cache/torch
export VIPE_RESULT_ROOT=/data/results/vipe_pano_pose

DATASET_FORMAT=omniroam_png \
JSON_PATH="${OMNIROAM_SPLIT_JSON}" \
IMAGE_BASE_DIR="${INTERIORGS_DATA_ROOT}" \
OUTPUT_ROOT_DIR="${VIPE_RESULT_ROOT}" \
RESOLUTION=1024 \
SPLIT_SUBSET=test \
INTERIORGS_FRAMES_SUBDIR=pano_camera0 \
INTERIORGS_MAX_FRAMES=800 \
INTERIORGS_FRAME_EXT=png \
GPU_IDS=0,1 \
MULTI_PROCESS_LAUNCH=1 \
START_CLIP_IDX=0 \
NUM_CLIPS_PER_GPU=0 \
LOG_TO_CONSOLE=true \
bash run_vipe_pano_pose.sh
```

脚本会根据 scene 数量自动分片，每个 GPU 启动一个独立进程。98 个 test scene 使用两张 GPU 时，默认每个进程处理 49 个。

正式运行前用下面的命令确认每张 GPU 有足够空闲显存：

```bash
nvidia-smi
```

输出位置为：

```text
${VIPE_RESULT_ROOT}/
├── logs/
│   ├── run_gpu0_start0.log
│   └── run_gpu1_start49.log
└── train_test_files_omniroam_png/
    ├── 0007_840137_poses.json
    ├── 0026_839976_poses.json
    └── ...
```

每个 pose 文件内容是：

```text
shape = [800, 4, 4]
类型 = C2W
pose[t][:3, :3] = R_c2w
pose[t][:3, 3]  = ViPE SLAM 世界中的相机中心
```

已有 pose 文件且没有传 `--overwrite` 时，脚本会跳过。若某些 scene 因 OOM 失败，清空 GPU 后直接重新执行同一条命令即可：成功文件会跳过，失败 scene 会再次尝试。

只有明确需要重新计算所有已有结果时才使用：

```bash
bash run_vipe_pano_pose.sh --overwrite
```

## 11. 检查完整推理结果

统计 pose 文件数：

```bash
export VIPE_POSE_ROOT="${VIPE_RESULT_ROOT}/train_test_files_omniroam_png"

find "${VIPE_POSE_ROOT}" \
  -maxdepth 1 \
  -type f \
  -name '*_poses.json' | wc -l
```

完整 test split 应为：

```text
98
```

检查所有 JSON 是否都是 `[800,4,4]` 且数值有限：

```bash
python - "${VIPE_POSE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
paths = sorted(root.glob("*_poses.json"))
bad = []

for path in paths:
    poses = np.asarray(json.loads(path.read_text(encoding="utf-8")))
    if poses.shape != (800, 4, 4) or not np.isfinite(poses).all():
        bad.append((path.name, poses.shape))

print("pose files:", len(paths))
print("invalid files:", len(bad))
for item in bad:
    print(item)

if bad:
    raise SystemExit(1)
PY
```

## 12. 计算单个 scene 的指标

推荐显式设置全部评测参数，不依赖 Python 默认值：

```bash
cd "${VIPE_REPO}"
conda activate vipe-panorama-cu124

export VIPE_POSE_ROOT="${VIPE_RESULT_ROOT}/train_test_files_omniroam_png"
export VIPE_EVAL_ROOT=/data/results/vipe_pano_pose_eval

VIDEO_ID=0007_840137 \
VIPE_POSE_ROOT="${VIPE_POSE_ROOT}" \
INTERIORGS_ROOT="${INTERIORGS_DATA_ROOT}" \
OUTPUT_ROOT="${VIPE_EVAL_ROOT}" \
FRAMES_SUBDIR=pano_camera0 \
FRAME_EXT=png \
FRAME_START=1 \
FRAME_COUNT=800 \
ALIGN_MODE=sim3 \
ROTATION_MODE=absolute \
GT_POSITION_SOURCE=location \
GT_ROTATION_SOURCE=interiorgs_erp \
WORLDSCORE_MODE=relative \
PREVIEW_ROWS=10 \
bash run_compare_vipe_omniroam_pose.sh
```

输出：

```text
${VIPE_EVAL_ROOT}/0007_840137/
├── 0007_840137_vipe_vs_gt.csv
└── 0007_840137_vipe_vs_gt_summary.json
```

正式报告的 WorldScore-style 主指标是：

```text
worldscore_rotation_mean_deg    越低越好
worldscore_translation_mean     越低越好
```

必须使用：

```text
WORLDSCORE_MODE=relative
```

不要将 `raw` 结果作为 ViPE SLAM 的最终指标，因为 ViPE SLAM 世界原点和 GT 世界原点不同。

以下字段是辅助诊断指标：

```text
translation_after              Sim3 对齐后的 ATE 类误差
rotation_error_deg             Sim3 对齐后的绝对旋转误差
translation_step_error         相邻帧平移误差
relative_rotation_step_error_deg
```

## 13. 批量计算所有成功 scene

当前 `run_compare_vipe_omniroam_pose.sh` 一次评测一个 scene。遍历所有成功生成的 pose：

```bash
cd "${VIPE_REPO}"
conda activate vipe-panorama-cu124

export VIPE_POSE_ROOT="${VIPE_RESULT_ROOT}/train_test_files_omniroam_png"
export INTERIORGS_ROOT="${INTERIORGS_DATA_ROOT}"
export OUTPUT_ROOT="${VIPE_EVAL_ROOT}"

export FRAME_START=1
export FRAME_COUNT=800
export ALIGN_MODE=sim3
export ROTATION_MODE=absolute
export GT_POSITION_SOURCE=location
export GT_ROTATION_SOURCE=interiorgs_erp
export WORLDSCORE_MODE=relative
export PREVIEW_ROWS=0

shopt -s nullglob
pose_paths=("${VIPE_POSE_ROOT}"/*_poses.json)

if [[ "${#pose_paths[@]}" -eq 0 ]]; then
    echo "No pose JSON files found in ${VIPE_POSE_ROOT}" >&2
    exit 1
fi

for pose_path in "${pose_paths[@]}"; do
    pose_name="$(basename "${pose_path}")"
    video_id="${pose_name%_poses.json}"
    echo "Evaluating ${video_id}"
    VIDEO_ID="${video_id}" bash run_compare_vipe_omniroam_pose.sh
done
```

评测是 CPU 计算，比 ViPE 推理快得多。

## 14. 汇总所有 scene 的主指标

下面的命令读取所有 summary，并计算 scene-level macro average：

```bash
python - "${VIPE_EVAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
paths = sorted(root.glob("*/*_summary.json"))
rows = []

for path in paths:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("worldscore_mode") != "relative":
        continue
    rows.append(
        (
            path.parent.name,
            float(summary["worldscore_rotation_mean_deg"]),
            float(summary["worldscore_translation_mean"]),
        )
    )

if not rows:
    raise SystemExit("No relative-mode summaries found")

rotation = np.asarray([row[1] for row in rows])
translation = np.asarray([row[2] for row in rows])

print("evaluated scenes:", len(rows))
print("worldscore_rotation_mean_deg macro average:", rotation.mean())
print("worldscore_translation_mean macro average:", translation.mean())
PY
```

同时报告成功覆盖率。例如只成功生成了 96/98 个 pose，就应该明确写出：

```text
evaluated scenes = 96 / 98
```

不要只报告成功子集均值而隐藏失败 scene。

## 15. SciPy 与 CVXPY 的复现问题

`compare_vipe_omniroam_pose.py` 求 WorldScore scalar scale 时按以下顺序选择求解器：

```text
安装了 cvxpy -> 使用 cvxpy
没有 cvxpy   -> 使用 scipy_minimize_scalar
```

仓库的 `envs/requirements-cu124.txt` 没有安装 CVXPY，因此按照本文创建的干净环境会使用：

```text
worldscore_scale_solver = scipy_minimize_scalar
```

历史基线中的 `0007_840137` 结果使用 SciPy 生成。两种求解器的数值差异通常极小，但严格跨机器复现时必须统一；本教程固定使用未安装 CVXPY 的干净环境。

建议新机器遵循以下规则：

1. 只安装 `envs/requirements-cu124.txt` 中锁定的依赖。
2. 不额外安装 CVXPY。
3. 在每个 summary 中检查：

```text
worldscore_scale_solver = scipy_minimize_scalar
worldscore_mode = relative
align_mode = sim3
rotation_mode = absolute
gt_position_source = location
gt_rotation_source = interiorgs_erp
```

CVXPY 不影响 ViPE pose 推理，只影响后续尺度求解。

## 16. 保存运行元数据

当前 pose JSON 本身只保存 `[T,4,4]` C2W 矩阵，没有保存代码版本、环境版本和启动参数。正式运行前建议创建输出目录并保存元数据：

```bash
mkdir -p "${VIPE_RESULT_ROOT}"

git rev-parse HEAD \
  > "${VIPE_RESULT_ROOT}/git_commit.txt"

conda list -n vipe-panorama-cu124 \
  > "${VIPE_RESULT_ROOT}/conda_list.txt"

python -m pip freeze \
  > "${VIPE_RESULT_ROOT}/pip_freeze.txt"

nvidia-smi \
  > "${VIPE_RESULT_ROOT}/nvidia_smi.txt"

nvcc --version \
  > "${VIPE_RESULT_ROOT}/nvcc_version.txt"

python -c "import torch; print(torch.__version__); print(torch.version.cuda)" \
  > "${VIPE_RESULT_ROOT}/torch_cuda_version.txt"
```

另外应保存实际使用的参数：

```text
JSON_PATH
IMAGE_BASE_DIR
RESOLUTION
SPLIT_SUBSET
INTERIORGS_MAX_FRAMES
GPU_IDS
ALIGN_MODE
ROTATION_MODE
GT_POSITION_SOURCE
GT_ROTATION_SOURCE
WORLDSCORE_MODE
```

CUDA 算子可能存在轻微数值非确定性，因此跨 GPU 型号运行时不保证 pose JSON 逐位完全一致，但环境、模型权重、代码和参数一致时，最终指标应当接近。

## 17. 常见问题

### 17.1 `nvcc: command not found`

按照第 5 节安装 `cuda-nvcc`，然后执行：

```bash
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
hash -r
```

### 17.2 `ModuleNotFoundError: No module named 'vipe_ext'`

说明自定义 CUDA 扩展尚未成功编译，或者运行时使用了另一个 Python 环境。检查：

```bash
which python
which nvcc
python -m pip show vipe
```

然后回到仓库根目录重新执行第 6 节安装命令。

### 17.3 CUDA/PyTorch 版本不匹配

确认三处都是 CUDA 12.4：

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pip show vipe
```

ViPE 包版本应包含类似：

```text
+pt26cu124
```

### 17.4 DROID 权重下载失败

离线复制 `droid.pth` 到：

```text
${TORCH_HOME}/hub/droid_slam/droid.pth
```

并核对第 7 节 SHA256。

### 17.5 CUDA out of memory

先检查是否有其他进程占用 GPU：

```bash
nvidia-smi
```

确保一个 ViPE worker 独占一张 GPU。显存碎片明显时可以设置：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

先使用 `INTERIORGS_MAX_FRAMES=20` 完成 smoke test。正式指标必须恢复为 800 帧。降低输入 `RESOLUTION` 会改变推理配置和结果，不应与 1024 分辨率结果混合汇总。

### 17.6 某些 scene 失败

不要立即使用 `--overwrite`。清理 GPU 占用后重新执行相同完整推理命令，已有结果会跳过，缺失结果会重试。

### 17.7 GT frame 缺失

确认 frame 编号从 1 开始：

```text
frame_0001.png ... frame_0800.png
```

评测参数必须对应：

```text
FRAME_START=1
FRAME_COUNT=800
FRAMES_SUBDIR=pano_camera0
FRAME_EXT=png
```

### 17.8 直接运行 Python 得到不同 rotation diagnostic

`run_compare_vipe_omniroam_pose.sh` 显式设置了：

```text
ROTATION_MODE=absolute
```

而直接运行 `compare_vipe_omniroam_pose.py` 时，Python argparse 的默认值是 `constant_offset`。为了复现当前结果，应使用本文命令显式传递所有参数，或者始终通过 shell 脚本运行。

## 18. 最终检查清单

开始正式运行前逐项确认：

- [ ] Git commit 已固定。
- [ ] 已激活 `vipe-panorama-cu124`。
- [ ] PyTorch 是 `2.6.0+cu124`。
- [ ] `torch.cuda.is_available()` 为 `True`。
- [ ] `nvcc` 是 CUDA 12.4。
- [ ] `import vipe_ext` 成功。
- [ ] `droid.pth` SHA256 正确。
- [ ] test split 包含 98 个 scene。
- [ ] 每个 scene 有 `frame_0001.png ... frame_0800.png`。
- [ ] 每个 scene 有 `transforms.json`。
- [ ] 20 帧 smoke test 成功。
- [ ] 正式推理使用 `RESOLUTION=1024` 和 800 帧。
- [ ] 每个 GPU 只运行一个 ViPE worker，且有足够空闲显存。
- [ ] 最终得到 98 个 `[800,4,4]` pose JSON，或明确记录失败 scene。
- [ ] 评测使用 `WORLDSCORE_MODE=relative`。
- [ ] 评测使用 `GT_POSITION_SOURCE=location`。
- [ ] 评测使用 `GT_ROTATION_SOURCE=interiorgs_erp`。
- [ ] summary 中的 scale solver 在所有机器上一致。
- [ ] 最终报告两个主指标和成功覆盖率。

最终主指标为：

```text
worldscore_rotation_mean_deg
worldscore_translation_mean
```

二者都是越低越好。
