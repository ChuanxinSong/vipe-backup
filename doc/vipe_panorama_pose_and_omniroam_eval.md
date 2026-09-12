# ViPE Panorama Pose Estimation And OmniRoam Evaluation

本文整理当前纯位姿推理脚本和 OmniRoam GT 对比脚本的处理逻辑。相关文件：

- `infer_vipe_panorama_pose.py`
- `run_vipe_pano_pose.sh`
- `compare_vipe_omniroam_pose.py`

## 1. 纯位姿推理入口

`infer_vipe_panorama_pose.py` 只运行 ViPE panorama SLAM 位姿估计，不运行 Unik3D、DepthCrafter、全景深度融合、深度可视化或深度保存。

它支持三种输入：

```text
pvdepth_json
omniroam_png
omniroam_h5
```

### pvdepth_json

参数含义：

```text
--json_path       pvdepth 嵌套 clip JSON
--image_base_dir  rgb_path 拼接根目录
```

输出：

```text
<output_root>/<json_stem>/<clip_name>_poses.json
```

### omniroam_png

参数含义：

```text
--json_path                    OmniRoam split JSON，例如 train_test_files.json
--image_base_dir                InteriorGS/OmniRoam PNG 根目录
--split_subset                  默认 test
--interiorgs_frames_subdir      默认 pano_camera0
--interiorgs_max_frames         默认 800
--interiorgs_frame_ext          默认 png
```

每个 `video_id` 被当作一个完整 clip。帧路径为：

```text
<image_base_dir>/<video_id>/<frames_subdir>/frame_XXXX.<ext>
```

默认读取：

```text
frame_0001.png ... frame_0800.png
```

输出：

```text
<output_root>/<split_json_stem>_omniroam_png/<video_id>_poses.json
```

### omniroam_h5

参数含义：

```text
--json_path       OmniRoam split JSON
--h5_data_root    H5 根目录
```

每个 H5 文件路径为：

```text
<h5_data_root>/<video_id>.h5
```

要求包含：

```text
frame_id
rgb
```

`frame_id` 用来映射 `1..interiorgs_max_frames` 到 H5 row，`rgb` 要是 `[N,H,W,3]` RGB uint8。

输出：

```text
<output_root>/<split_json_stem>_omniroam_h5/<video_id>_poses.json
```

## 2. 推理流程

每个 clip 的处理步骤：

1. 展开输入 JSON / split JSON，得到统一 clip descriptor。
2. 检查重复 clip id，避免输出覆盖。
3. 如果目标 `<clip_name>_poses.json` 已存在且没有 `--overwrite`，直接跳过。
4. 严格检查每帧图像或 H5 row：
   - PNG 缺失或无法 decode，当前 clip 失败。
   - H5 缺少 `frame_id` 或 `rgb`，当前 clip 失败。
   - H5 缺少目标 frame id，当前 clip 失败。
5. 全景帧 resize 到：

```text
height = resolution / 2
width  = resolution
```

6. 读取 `configs/pipeline/panorama.yaml`，构建虚拟 pinhole 视角：

```text
4 个水平视角
1 个 bottom 视角
top=false
```

7. 合并 `configs/slam/default.yaml` 和 panorama SLAM 配置，并强制：

```text
visualize = false
optimize_intrinsics = false
keyframe_depth = None
```

8. 运行：

```python
SLAMSystem.run(slam_streams, rig=rig_se3)
```

9. 保存：

```python
slam_output.trajectory.matrix()
```

保存内容是 ViPE 输出的逐帧 C2W 矩阵：

```text
shape = [T, 4, 4]
```

10. 使用临时文件和 `os.replace()` 原子写入，避免中断产生半写文件。

单个 clip 失败时记录错误并继续处理后续 clip。只要存在任意失败，进程最终返回非零状态。

## 3. 输出位姿格式

ViPE 输出：

```text
*_poses.json = C2W 4x4 trajectory
```

含义：

```text
pose[t][:3, :3] = R_c2w
pose[t][:3, 3]  = camera center C in ViPE SLAM world
```

代码依据：

```python
SLAMSystem.run() 返回 trajectory=filled_return.poses.inv()
FilledReturn.poses 注释是 Inverse of c2w
```

因此它不是：

```text
不是 OpenCV W2C [R|t]
不是 Control-Camera entry
不是 H5 pose=[x,y,z]
不是 transforms.json 的 transform_matrix
```

如果要转成 OpenCV W2C：

```python
w2c = np.linalg.inv(c2w)
R = w2c[:3, :3]
t = w2c[:3, 3]
```

## 4. Shell 启动脚本

`run_vipe_pano_pose.sh` 默认跑 OmniRoam PNG：

```bash
DATASET_FORMAT=omniroam_png
JSON_PATH=OmniRoam/configs/train_test_files.json
IMAGE_BASE_DIR=/data1/songcx/dataset/interiogs_render
OUTPUT_ROOT_DIR=carla_benchmark_results/vipe_pano_pose
RESOLUTION=1024
SPLIT_SUBSET=test
INTERIORGS_MAX_FRAMES=800
GPU_ID=0
```

运行：

```bash
bash run_vipe_pano_pose.sh
```

改 GPU：

```bash
GPU_ID=6 bash run_vipe_pano_pose.sh
```

降低分辨率加速：

```bash
RESOLUTION=768 bash run_vipe_pano_pose.sh
```

只跑短序列测试：

```bash
INTERIORGS_MAX_FRAMES=100 bash run_vipe_pano_pose.sh
```

重跑已存在结果：

```bash
bash run_vipe_pano_pose.sh --overwrite
```

切换 H5：

```bash
DATASET_FORMAT=omniroam_h5 \
H5_DATA_ROOT=/path/to/interiogs_render_h5 \
bash run_vipe_pano_pose.sh
```

切回 pvdepth：

```bash
DATASET_FORMAT=pvdepth_json \
JSON_PATH=/path/to/setting.json \
IMAGE_BASE_DIR=/path/to/pvdepth \
bash run_vipe_pano_pose.sh
```

## 5. OmniRoam / InteriorGS 坐标轴

根据 `OmniRoam/doc/pose_i2v_h5_ray_alignment_findings.md`，这批 InteriorGS/OmniRoam 数据源头的世界坐标是：

```text
world X = right
world Y = forward
world Z = up
```

`transforms.json` 中：

```text
location = path.json pos = H5 pose = [x, y, z]
R = I
t = -location
```

所以位置 GT 应优先使用：

```text
transforms.json["location"]
```

而不是 `transform_matrix`。

ViPE/OpenCV camera-local 轴可以按：

```text
camera +X = right
camera +Y = down
camera +Z = forward
```

因此在 OmniRoam InteriorGS world 中，默认 GT C2W rotation 使用：

```text
camera +X/right   -> world +X/right
camera +Y/down    -> world -Z/up
camera +Z/forward -> world +Y/forward
```

矩阵列为 camera axes expressed in world：

```python
R_c2w = np.array([
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
    [0.0, -1.0, 0.0],
])
```

`compare_vipe_omniroam_pose.py` 中对应：

```text
--gt_rotation_source interiorgs_erp
```

这是默认值。

## 6. GT 对比脚本

`compare_vipe_omniroam_pose.py` 用来比较 ViPE C2W 和 OmniRoam GT。

推荐命令：

```bash
python compare_vipe_omniroam_pose.py \
  --vipe_poses carla_benchmark_results/vipe_pano_pose/train_test_files_omniroam_png/0007_840137_poses.json \
  --gt_transforms_json /data1/songcx/dataset/interiogs_render/0007_840137/transforms.json \
  --output_csv /tmp/0007_840137_vipe_vs_gt.csv \
  --output_summary_json /tmp/0007_840137_vipe_vs_gt_summary.json
```

默认参数：

```text
--align_mode sim3
--rotation_mode absolute
--gt_position_source location
--gt_rotation_source interiorgs_erp
--worldscore_mode relative
```

### WorldScore-style 主指标

当前脚本使用 ViPE 输出的 C2W pose 替代 WorldScore 源码中的 DROID-SLAM pose，只复用 WorldScore 的 pose error 计算方式。

WorldScore 源码实际返回的是：

```text
(mean rotation error in degrees, mean translation error)
```

rotation error 按源码公式：

```text
e_R = acos((trace(R_pred * R_gt.T) - 1) / 2) * 180 / pi
```

translation error 只允许一个 scalar scale：

```text
s = argmin_s sum_i ||t_gt_i - s * t_pred_i||_2
e_t = ||t_gt_i - s * t_pred_i||_2
```

由于 ViPE / SLAM 世界原点任意，默认使用相对首帧模式：

```text
--worldscore_mode relative
```

即：

```text
t_gt   = t_gt   - t_gt[0]
t_pred = t_pred - t_pred[0]
```

如果需要更贴近 WorldScore 源码的 raw 坐标口径，可以使用：

```text
--worldscore_mode raw
```

ViPE pose 在进入 WorldScore-style 指标前会先应用固定坐标轴转换：

```python
R_pred_ws = R_gt[0] @ R_vipe
t_pred_ws = R_gt[0] @ t_vipe
```

这一步对应 WorldScore 源码中对 DROID 输出做固定坐标系转换的作用；它不是 Sim3 拟合。

### align_mode

`none`：

```text
不对齐。只适合检查已经同坐标系、同尺度的轨迹。
```

`scale`：

```text
只估计尺度和平移，不估计全局旋转。
```

`sim3`：

```text
估计尺度、全局旋转、全局平移。
```

ViPE 是单目 SLAM 原始尺度，且世界坐标由 SLAM 自己建立，所以默认应该使用 `sim3`。

`align_mode` 只影响 legacy diagnostic 指标，不影响 WorldScore-style 主指标。

### rotation_mode

`absolute`：

```text
直接比较对齐后的 C2W rotation。
```

配合 `--gt_rotation_source interiorgs_erp` 时，这个指标已经可以直接解释。

`relative`：

```text
比较相对第一帧的 rotation change。
```

适合排除固定相机轴差异，只看旋转运动是否一致。

`constant_offset`：

```text
在 absolute 基础上额外拟合一个固定 camera-axis rotation offset。
```

适合排查仍然存在固定坐标轴约定差异的情况。

`rotation_mode` 只影响 legacy diagnostic 指标，不影响 WorldScore-style 主指标。

## 7. 输出指标

脚本输出 summary 表，并保存：

```text
*_vs_gt.csv
*_vs_gt_summary.json
```

主要指标：

```text
worldscore_rotation_mean_deg
worldscore_translation_mean
worldscore_scale
worldscore_rotation_error_deg
worldscore_translation_error
translation_rmse_after
translation_mean_after
translation_max_after
translation_step_rmse
rotation_rmse_deg
rotation_mean_deg
rotation_max_deg
relative_rotation_step_rmse_deg
similarity_scale
gt_path_length
vipe_path_length
aligned_path_length
```

CSV 中逐帧字段：

```text
frame_id
gt_x, gt_y, gt_z
vipe_x, vipe_y, vipe_z
aligned_x, aligned_y, aligned_z
translation_error_before
translation_error_after
rotation_error_deg
relative_rotation_step_error_deg
worldscore_rotation_error_deg
worldscore_translation_error
worldscore_scaled_pred_rel_x/y/z
gt_rel_x/y/z
pred_rel_x/y/z
gt_step
aligned_step
translation_step_error
```

## 8. 当前 0007_840137 示例结果

使用：

```text
gt_rotation_source = interiorgs_erp
rotation_mode = absolute
align_mode = sim3
```

得到：

```text
worldscore_rotation_mean_deg    = 0.204117928
worldscore_translation_mean     = 0.011167790
worldscore_scale                = 1.824170196
translation_rmse_after          = 0.003185252
translation_mean_after          = 0.002933347
translation_max_after           = 0.006139493
translation_step_rmse           = 0.000283238
rotation_rmse_deg               = 0.219960398
rotation_mean_deg               = 0.217345609
rotation_max_deg                = 0.279241942
relative_rotation_step_rmse_deg = 0.008000838
similarity_scale                = 1.825472226
```

这说明该 clip 中，ViPE 轨迹在 Sim3 对齐后和平移 GT 非常接近；按 InteriorGS ERP 坐标轴构造 rotation GT 后，绝对旋转误差也在约 `0.2` 度量级。

注意：`translation_rmse_after` 是 Sim3 diagnostic，不是 WorldScore-style translation error；主指标应看 `worldscore_translation_mean`。

## 9. WorldScore 指标解释

Camera controllability error 是误差指标，越低越好：

```text
e_theta: rotation angular deviation, lower is better
e_t: scale-invariant translation distance, lower is better
e_camera = sqrt(e_theta * e_t), lower is better
```

WorldScore 源码实际返回并归一化的是：

```text
(mean rotation error, mean translation error)
```

并在配置中标记：

```text
higher_is_better = [False, False]
```

当前脚本字段对应关系：

```text
worldscore_rotation_error_deg  -> per-frame e_theta in degrees
worldscore_rotation_mean_deg   -> mean(e_theta)
worldscore_translation_error   -> per-frame e_t
worldscore_translation_mean    -> mean(e_t)
worldscore_scale               -> S2 中的 scalar scale s
```

默认 `WORLDSCORE_MODE=relative` 会先减首帧：

```text
t_gt   = t_gt   - t_gt[0]
t_pred = t_pred - t_pred[0]
```

然后按 WorldScore 源码形式求：

```text
s = argmin_s sum_i ||t_gt_i - s * t_pred_i||_2
e_t_i = ||t_gt_i - s * t_pred_i||_2
```

这是为了适配 ViPE / SLAM 世界原点任意的问题。`WORLDSCORE_MODE=raw` 不减首帧，更接近源码 raw 坐标口径，但对 ViPE 输出通常不适合作为最终指标。

## 10. 注意事项

- 不要用 `transforms.json["transform_matrix"]` 作为 OpenCV 外参或 GT camera center。
- 对 OmniRoam / InteriorGS，位置 GT 优先使用 `location` 或 H5 `pose` 原始 `[x,y,z]`。
- ViPE 输出是 C2W；Control-Camera 需要 W2C 时必须取逆。
- ViPE 原始尺度任意，跨数据集比较必须做 scale 或 Sim3 对齐。
- `SLAMSystem.run()` 返回前仍会内部提取一次 `slam_map`，当前纯位姿脚本不会使用或保存它。
