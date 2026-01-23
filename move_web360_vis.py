import os
import shutil

# 原始路径
src_root = "/home/user/songcx/code/vipe/web360_results"
# 目标路径
dst_root = os.path.join(src_root, "web360_results_video")

# 如果目标目录不存在，创建它
os.makedirs(dst_root, exist_ok=True)

# 遍历原始路径下的所有文件夹
for folder_name in os.listdir(src_root):
    folder_path = os.path.join(src_root, folder_name)
    
    # 确保是文件夹且不是目标文件夹
    if os.path.isdir(folder_path) and folder_name != "web360_results_video":
        vis_path = os.path.join(folder_path, f"{folder_name}_vis.mp4")
        
        # 检查文件是否存在
        if os.path.isfile(vis_path):
            new_filename = f"{folder_name}_vis.mp4"
            dst_path = os.path.join(dst_root, new_filename)
            
            # 移动并重命名
            shutil.move(vis_path, dst_path)
            print(f"已移动: {vis_path} -> {dst_path}")
        else:
            print(f"跳过: {folder_path} 中没有 vis.mp4")

print("所有视频已处理完成。")