# ViPE 全景视频 Unik3D 处理流程

本文档梳理 `run_vipe_pano_unik3d.sh` 相关脚本的调用链，以及 ViPE 对全景视频序列进行 SLAM、Unik3D 深度估计和尺度融合的整体逻辑。

## 1. 整体调用链

```text
run_vipe_pano_unik3d.sh
  └─ run_vipe_pano_all.py
      └─ infer_vipe_panorama.py
          ├─ 从 JSON 加载全景图像序列
          ├─ 将全景图投影为多个虚拟针孔相机视角
          ├─ 使用多视角 ViPE SLAM 估计轨迹和地图
          ├─ 使用 Unik3D 逐帧估计全景深度
          ├─ 将 Unik3D 深度缩放到 SLAM 地图尺度
          └─ 保存深度、有效掩码和可视化视频
```

相关文件：

- `run_vipe_pano_unik3d.sh`
- `run_vipe_pano_all.py`
- `infer_vipe_panorama.py`
- `configs/pipeline/panorama.yaml`
- `configs/slam/default.yaml`
- `vipe/pipeline/panorama.py`
- `vipe/pipeline/processors.py`
- `vipe/slam/system.py`
- `vipe/priors/depth/unik3d/__init__.py`

## 2. 启动脚本

`run_vipe_pano_unik3d.sh` 设置运行参数：

```bash
OUTPUT_ROOT_DIR="carla_benchmark_results/vipe_pano_unik3d"
IMAGE_BASE_DIR="/workspace1/songcx/dataset/pvdepth"
RESOLUTION=1024

CUDA_VISIBLE_DEVICES=6 python run_vipe_pano_all.py \
    --output_root_dir "${OUTPUT_ROOT_DIR}" \
    --resolution "${RESOLUTION}" \
    --image_base_dir "${IMAGE_BASE_DIR}" \
    --depth_method "unik3d"
```

主要参数：

| 参数 | 含义 |
| --- | --- |
| `OUTPUT_ROOT_DIR` | 所有推理结果的根目录 |
| `IMAGE_BASE_DIR` | JSON 中相对图像路径对应的数据集根目录 |
| `RESOLUTION` | 全景图宽度，高度固定为宽度的一半 |
| `depth_method=unik3d` | 使用 Unik3D 生成稠密全景深度 |
| `CUDA_VISIBLE_DEVICES=6` | 使用物理 GPU 6 |

## 3. 批量处理 JSON

`run_vipe_pano_all.py` 中通过 `JSON_FILES` 硬编码需要处理的测试配置：

```python
JSON_FILES = [
    "/workspace1/songcx/dataset/pvdepth/test_benchmark/setting_dynamic_fps02_len50.json",
]
```

脚本会依次为每个 JSON 启动独立子进程：

```text
python infer_vipe_panorama.py --json_path <json_path> <其他透传参数>
```

单个 JSON 大致按以下层级组织：

```text
town
  └─ path
      └─ clip_name
          └─ frames
              └─ rgb_path
```

`infer_vipe_panorama.py` 会将所有 clip 展平。后续主要使用：

- `clip_name`：输出文件名前缀。
- `frames[*].rgb_path`：每帧全景图相对路径。

## 4. 全景图像流

`infer_vipe_panorama.py` 中的 `SimpleJsonVideoStream` 将 JSON 图像序列适配为 ViPE 的 `VideoStream`。

每帧处理流程：

1. 使用 `IMAGE_BASE_DIR` 和 `rgb_path` 拼接完整图像路径。
2. 使用 OpenCV 读取图片。
3. Resize 为 `(RESOLUTION, RESOLUTION / 2)`。
4. 将 BGR 转换为 RGB。
5. 转换为 `float32`，归一化到 `[0, 1]`。
6. 将相机类型标记为 `CameraType.PANORAMA`。

当 `RESOLUTION=1024` 时，输入 ViPE 的全景图尺寸为：

```text
宽度 1024，高度 512
```

如果图片不存在或读取失败，脚本会使用纯黑帧代替，不会立即终止任务。

随后通过 `CachedVideoStream` 缓存帧，避免多个虚拟相机流重复读取同一张全景图。

## 5. 构造虚拟多相机 Rig

ViPE SLAM 主要处理针孔相机图像，因此脚本先将等距柱状全景图投影为多个虚拟针孔相机视角。

虚拟相机配置来自 `configs/pipeline/panorama.yaml`：

```yaml
virtual:
  height: 512
  fovx: 100.0
  fovy: 100.0
  num_views: 4
  top: false
  bottom: true
```

当前 `infer_vipe_panorama.py` 没有直接使用配置中的 `virtual.height`，而是根据全景高度计算：

```python
pano_height = args.resolution // 2
virtual_height = pano_height // 2
```

当输入全景图为 `1024 x 512` 时，每个虚拟视角约为 `256 x 256`。

当前实际生成 5 个虚拟视角：

```text
水平视角：0°、90°、180°、270°
底部视角：-90°
```

由于 `top: false`，顶部视角不会生成。源码中“生成 6 面视角”的注释与当前配置不一致。

`EquirectProjectionProcessor` 使用预计算采样网格和 `torch.nn.functional.grid_sample`，将全景图投影到每个针孔视角。

## 6. 多视角 ViPE SLAM

虚拟相机图像流和固定 rig 旋转关系被传入：

```python
slam_output = SLAMSystem(...).run(slam_streams, rig=rig_se3)
```

主要过程：

```text
虚拟针孔相机流
  └─ StandardResizeStreamProcessor
      └─ DROID 网络提取特征
          └─ MotionFilter 选择关键帧
              └─ Frontend 局部优化
                  └─ Backend 全局 Bundle Adjustment
                      └─ InnerFiller 补全非关键帧
                          └─ 提取 SLAM 点云地图
```

SLAM 输出包括：

- `trajectory`：所有原始帧的相机轨迹。
- `intrinsics`：虚拟相机内参。
- `rig`：多个虚拟相机之间的固定变换。
- `slam_map`：由关键帧视差和位姿生成的过滤后点云地图。

当前全景 SLAM 关键配置：

```yaml
slam:
  buffer: 256
  optimize_intrinsics: false
  keyframe_depth: null
  map_filter_thresh: 0.01
```

默认配置中还启用了：

```yaml
cross_view: true
```

因此多个虚拟视角会共同参与 SLAM 优化。

### 6.1 SLAM 尺度

当前配置强制设置：

```python
slam_cfg.keyframe_depth = None
```

这意味着 SLAM 没有使用外部 metric depth 固定全局尺度。SLAM 输出的轨迹和地图保持单目 SLAM 的任意尺度。

## 7. Unik3D 深度估计

`MergedPanoramaVideoStream` 根据 `pano_depth_method="unik3d"` 加载：

```python
UniK3D.from_pretrained("lpiccinelli/unik3d-vitl")
```

Unik3D 使用球面相机模型处理完整的全景图，输出 spherical distance map。

当前流程中，每帧完整的 `1024 x 512` 全景图会独立送入 Unik3D：

```text
全景 RGB 帧 -> Unik3D -> 单帧稠密 distance map
```

Unik3D 并未作为视频模型运行，因此其原始预测可能存在跨帧尺度变化或深度抖动。

## 8. Unik3D 与 SLAM 地图尺度对齐

最终保存的深度不是直接使用 Unik3D 原始输出，而是逐帧缩放到 ViPE SLAM 地图的尺度。

对于每一帧：

1. 获取 ViPE SLAM 全局点云。
2. 使用当前帧位姿将点云转换到当前相机坐标系。
3. 将点云投影到当前全景图。
4. 生成稀疏的 SLAM target distance map。
5. 在存在有效 SLAM 投影的位置计算缩放系数。

缩放系数计算逻辑：

```python
inv_scale = median(unik3d_distance / slam_distance)
```

最终输出深度：

```python
final_distance = unik3d_distance / inv_scale
```

因此最终深度具有以下特点：

- 稠密空间结构主要来自 Unik3D。
- 每帧整体尺度跟随 ViPE SLAM 地图。
- 只进行乘法 scale 对齐，不估计 shift。
- 没有直接融合 SLAM 稀疏深度值，只使用它们估计尺度。

当有效 SLAM 投影像素不足全图的 5% 时，脚本会复用上一帧的 `inv_scale`。如果第一帧也无法估计，初始值为 `1.0`。

### 8.1 “Metric Depth”的实际含义

代码将结果写入 `frame.metric_depth`，但当前 SLAM 没有 metric depth 约束，其全局尺度是任意的。

所以当前最终结果更准确地描述为：

```text
Unik3D 稠密全景深度形状 + ViPE SLAM 全局尺度
```

不能仅根据变量名认定最终 `distance.npy` 一定以米为单位。

## 9. 输出结果

输出目录由 JSON 文件名和运行参数共同决定：

```text
<OUTPUT_ROOT_DIR>/
└─ town0210_<resolution>_<setting>_<fps>_<length>_<depth_method>/
```

当前示例对应：

```text
carla_benchmark_results/vipe_pano_unik3d/
└─ town0210_1024_dynamic_fps02_len50_unik3d/
    ├─ <clip_name>_distance.npy
    ├─ <clip_name>_distance_valid_mask.npy
    └─ <clip_name>_vis.mp4
```

### 9.1 Distance

```text
文件：<clip_name>_distance.npy
Shape：[T, H, W]
示例：[T, 512, 1024]
```

内容为经过 SLAM 尺度对齐后的 Unik3D 全景 spherical distance。

### 9.2 Valid Mask

```text
文件：<clip_name>_distance_valid_mask.npy
Shape：[T, H, W]
类型：uint8
```

有效像素条件：

```python
np.isfinite(depth) & (depth > 0)
```

### 9.3 可视化视频

```text
文件：<clip_name>_vis.mp4
```

每帧由上下两部分拼接：

```text
上半部分：原始 RGB 全景图
下半部分：深度伪彩色图
```

可视化视频固定使用 `10 FPS` 保存。

当前脚本没有保存 SLAM pose、虚拟相机内参和逐帧深度缩放系数。

## 10. 完整数据流

```text
JSON 中的全景图片序列
        │
        ▼
读取并 Resize 为 1024 x 512
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
投影为 5 路虚拟针孔相机流          完整全景 RGB 帧
        │                              │
        ▼                              ▼
多视角 ViPE SLAM                 Unik3D 单帧深度
        │                              │
        ▼                              │
轨迹 + SLAM 点云地图                  │
        │                              │
        └──── 将 SLAM 地图投影到全景图 ─┘
                       │
                       ▼
             估计逐帧深度缩放系数
                       │
                       ▼
              Unik3D 深度 / scale
                       │
                       ▼
       distance.npy + valid_mask.npy + vis.mp4
```

## 11. 当前实现中的注意事项

1. `run_vipe_pano_all.py` 中的 JSON 列表是硬编码的，shell 参数不能直接控制数据集列表。
2. `fps02` 等字符串只影响输出目录名，`SimpleJsonVideoStream` 和可视化视频仍固定使用 10 FPS。
3. 当前实际使用 5 个虚拟视角，而不是注释中描述的 6 个视角。
4. Unik3D 是逐帧推理，没有显式的视频时序一致性约束。
5. 最终深度尺度跟随单目 ViPE SLAM，不保证是真实米制尺度。
6. 只要 `distance.npy` 和 `distance_valid_mask.npy` 已存在，clip 就会被跳过，即使可视化视频不存在。
7. 当前没有保存 SLAM pose、内参和逐帧 scale，后续无法仅依赖输出文件完整复现融合过程。
8. 图片读取失败时会使用黑帧继续运行，这可能影响 SLAM，但不会明显暴露为任务失败。

