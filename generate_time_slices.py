import cv2
import os
import numpy as np
from tqdm import tqdm

# ================= 配置路径 =================
LIST_FILE = "/home/user/songcx/code/vipe/web360_infer.txt"
VIDEO_ROOT = "/home/user/songcx/code/vipe/web360_results/web360_results_video"
OUTPUT_ROOT = "/home/user/songcx/code/vipe/web360_results/time_slicing_vis"

# 切片间隔 (像素)
SLICE_INTERVAL = 50

# 【新增配置】可视化的横向拉伸倍数
# 原本100帧=100像素宽，拉伸5倍变成500像素宽，方便观察
VIS_WIDTH_SCALE = 5 

# ================= 主逻辑 =================

def process_video(video_name_in_list):
    # 1. 构造文件名和路径
    base_name = os.path.splitext(video_name_in_list.strip())[0]
    vis_filename = f"{base_name}_vis.mp4"
    video_path = os.path.join(VIDEO_ROOT, vis_filename)

    if not os.path.exists(video_path):
        # print(f"[Warning] Video not found: {video_path}")
        return

    # 2. 建立输出文件夹
    save_dir = os.path.join(OUTPUT_ROOT, base_name)
    os.makedirs(save_dir, exist_ok=True)

    # 3. 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {video_path}")
        return

    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 确定要切片的所有 x 坐标
    x_positions = list(range(50, width, SLICE_INTERVAL))

    # 初始化数据容器
    slice_data = {x: [] for x in x_positions}
    
    # 读取第一帧用于画线参考
    ret, first_frame = cap.read()
    if not ret:
        return

    # 放入第一帧数据
    for x in x_positions:
        slice_data[x].append(first_frame[:, x, :])

    # 4. 遍历提取
    for _ in range(total_frames - 1):
        ret, frame = cap.read()
        if not ret:
            break
        for x in x_positions:
            slice_data[x].append(frame[:, x, :])

    cap.release()

    # 5. 生成并保存结果
    # print(f"Saving results for {base_name}...")
    
    for x in x_positions:
        # --- A. 生成时间切片图 ---
        # 原始形状: (Height, Time, 3)
        raw_slice = np.stack(slice_data[x], axis=1)
        
        # 【修改点】执行拉伸
        h, w, c = raw_slice.shape
        new_w = w * VIS_WIDTH_SCALE
        # 使用 INTER_NEAREST 保持像素原本的硬度，不进行平滑欺骗
        resized_slice = cv2.resize(raw_slice, (new_w, h), interpolation=cv2.INTER_NEAREST)
        
        slice_save_path = os.path.join(save_dir, f"pos_{x}_slice.jpg")
        cv2.imwrite(slice_save_path, resized_slice)

        # --- B. 生成带红线的参考图 ---
        ref_img = first_frame.copy()
        # 画一条红线，加宽到 3-5 像素以便缩略图能看清
        cv2.line(ref_img, (x, 0), (x, height), (0, 0, 255), thickness=5)
        
        ref_save_path = os.path.join(save_dir, f"pos_{x}_ref.jpg")
        cv2.imwrite(ref_save_path, ref_img)

def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    with open(LIST_FILE, 'r') as f:
        video_list = [line.strip() for line in f if line.strip()]

    print(f"Found {len(video_list)} videos in task list.")

    for video_name in tqdm(video_list, desc="Processing"):
        process_video(video_name)

    print("All tasks finished.")

if __name__ == "__main__":
    main()