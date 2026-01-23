import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil

# ================= 配置区域 =================
UNIK3D_DIR = "/datawm/songcx/results/unik3d/web360_results/web360_results_video"
VIPE_DIR   = "/data3/songcx/results/vipe/web360_results/web360_results_video"
OUTPUT_ROOT = "/home/user/songcx/code/vipe/web360_results/compare_strict" # 新路径

SLICE_INTERVAL = 50    # 稍微密集一点
VIS_WIDTH_SCALE = 5    
TOP_K = 5              

# ===========================================

def normalize_slice(slice_img):
    """
    强制归一化：消除因对比度不同导致的“伪平滑”。
    将像素值拉伸到 0-255 范围。
    """
    if slice_img.max() == slice_img.min():
        return slice_img
    norm_img = cv2.normalize(slice_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm_img

def calculate_scores(slice_u, slice_v):
    """
    计算复杂评分：
    1. 必须是空间上有纹理的地方 (Spatial Variance)
    2. Unik3d 必须有剧烈抖动 (Max Temporal Gradient)
    3. Vipe 必须相对平稳
    """
    # 1. 归一化 (避免 Vipe 因为颜色淡而占便宜)
    u_norm = normalize_slice(slice_u)
    v_norm = normalize_slice(slice_v)

    # 转灰度
    g_u = cv2.cvtColor(u_norm, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_v = cv2.cvtColor(v_norm, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2. 计算时序梯度 (抖动)
    # shape: (Height, Time-1)
    temp_grad_u = np.abs(np.diff(g_u, axis=1))
    temp_grad_v = np.abs(np.diff(g_v, axis=1))

    # 3. 计算空间梯度 (判断是不是物体边缘)
    # 取第一帧或平均帧来计算这一列像素在垂直方向的变化
    # axis=0 是沿高度方向
    spatial_variance = np.std(g_u, axis=0).mean() # 这一列像素的垂直标准差

    # 如果这是一面平墙（标准差很小），直接忽略，不论修得多好都没意义
    if spatial_variance < 30.0: 
        return 0.0

    # 4. 核心差异计算
    # 我们寻找 Unik3d 抖动极大，而 Vipe 抖动极小的地方
    # 使用 ReLU 逻辑：只关心 Vipe 变好的地方，不管变差的地方
    improvement = np.maximum(0, temp_grad_u - temp_grad_v)
    
    # 5. 最终得分 = 改进量 * 空间重要性
    # 只有在边缘处修复了抖动，才是高分
    final_score = np.sum(improvement) * np.log1p(spatial_variance)

    return final_score

def process_comparison(filename):
    path_u = os.path.join(UNIK3D_DIR, filename)
    path_v = os.path.join(VIPE_DIR, filename)

    if not os.path.exists(path_u) or not os.path.exists(path_v): return

    cap_u = cv2.VideoCapture(path_u)
    cap_v = cv2.VideoCapture(path_v)
    
    if not cap_u.isOpened() or not cap_v.isOpened(): return
    
    ret_u, frame_u0 = cap_u.read()
    ret_v, frame_v0 = cap_v.read()
    if not ret_u or not ret_v: return
    
    # --- 裁剪逻辑 (保持不变) ---
    h_v, w_v, _ = frame_v0.shape
    frame_v0 = frame_v0[h_v//2:, :, :]
    
    crop_unik3d = False
    if frame_u0.shape[0] == 2 * frame_v0.shape[0]:
        crop_unik3d = True
        h_u, w_u, _ = frame_u0.shape
        frame_u0 = frame_u0[h_u//2:, :, :]
    
    if frame_u0.shape != frame_v0.shape: return

    height, width, _ = frame_u0.shape
    frames_u = int(cap_u.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_v = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))
    min_frames = min(frames_u, frames_v)

    x_positions = list(range(50, width, SLICE_INTERVAL))
    slices_u = {x: [] for x in x_positions}
    slices_v = {x: [] for x in x_positions}

    # 填充第一帧
    for x in x_positions:
        slices_u[x].append(frame_u0[:, x, :])
        slices_v[x].append(frame_v0[:, x, :])

    # 遍历后续帧
    for _ in range(min_frames - 1):
        ret_u, frame_u = cap_u.read()
        ret_v, frame_v = cap_v.read()
        if not ret_u or not ret_v: break
        
        # 裁剪
        frame_v = frame_v[frame_v.shape[0]//2:, :, :]
        if crop_unik3d:
             frame_u = frame_u[frame_u.shape[0]//2:, :, :]
        
        if frame_u.shape != frame_v.shape: break

        for x in x_positions:
            slices_u[x].append(frame_u[:, x, :])
            slices_v[x].append(frame_v[:, x, :])

    cap_u.release()
    cap_v.release()

    scores = [] 

    # --- 计算分数 ---
    for x in x_positions:
        img_u = np.stack(slices_u[x], axis=1)
        img_v = np.stack(slices_v[x], axis=1)

        # 使用新的严格打分逻辑
        score = calculate_scores(img_u, img_v)

        if score > 1000: # 提高门槛
            scores.append((score, x, img_u, img_v))

    scores.sort(key=lambda item: item[0], reverse=True)

    if len(scores) > 0:
        base_name = os.path.splitext(filename)[0]
        save_dir = os.path.join(OUTPUT_ROOT, base_name)
        os.makedirs(save_dir, exist_ok=True)
        
        ref_img = frame_u0.copy()
        
        for i in range(min(TOP_K, len(scores))):
            score, x, s_u, s_v = scores[i]
            
            h, w, c = s_u.shape
            new_w = w * VIS_WIDTH_SCALE
            # 这里的插值用 Nearest 保持锯齿原貌
            vis_u = cv2.resize(s_u, (new_w, h), interpolation=cv2.INTER_NEAREST)
            vis_v = cv2.resize(s_v, (new_w, h), interpolation=cv2.INTER_NEAREST)
            
            combined = np.hstack([vis_u, vis_v])
            
            save_name = f"rank{i+1}_score{int(score)}_pos{x}.jpg"
            cv2.imwrite(os.path.join(save_dir, save_name), combined)
            
            color = (0, 0, 255) if i == 0 else (0, 255, 255)
            cv2.line(ref_img, (x, 0), (x, height), color, 3)
            # 加上 Score 方便调试
            cv2.putText(ref_img, f"R{i+1}", (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imwrite(os.path.join(save_dir, "reference_positions.jpg"), ref_img)

def main():
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT)
    
    if not os.path.exists(VIPE_DIR): return

    all_files = os.listdir(VIPE_DIR)
    video_files = sorted([f for f in all_files if f.endswith('_vis.mp4')])
    
    print(f"Strict filtering on {len(video_files)} videos...")
    
    for f in tqdm(video_files):
        process_comparison(f)

    print(f"Results saved to {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()