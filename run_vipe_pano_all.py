import subprocess
import argparse
import sys
import os

# 请根据你的实际情况修改这里的列表 (保持与 Unik3D 的测试集一致)
JSON_FILES = [
    "/workspace1/songcx/dataset/pvdepth/test_benchmark/setting_dynamic_fps02_len50.json",
    # "/workspace1/songcx/dataset/pvdepth/test_benchmark/setting_dynamic_fps10_len90.json",
    # "/workspace1/songcx/dataset/pvdepth/test_benchmark/setting_dynamic_fps20_len110.json",
    # ... 其他配置文件
]

SCRIPT_NAME = "infer_vipe_panorama.py"

def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # 接收 run_vipe_pano_all.py 的参数并透传给 infer 脚本
    parser.add_argument("--json_path", help="Ignored") 
    _, passthrough = parser.parse_known_args()

    for i, json_path in enumerate(JSON_FILES):
        print(f"\n>>> [{i+1}/{len(JSON_FILES)}] Processing: {json_path}")
        
        if not os.path.exists(json_path):
            print(f"File not found: {json_path}")
            continue

        # 构造命令
        cmd = [sys.executable, SCRIPT_NAME, "--json_path", json_path] + passthrough
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Error processing: {json_path}")
            # sys.exit(1) # 可选：出错停止

if __name__ == "__main__":
    main()