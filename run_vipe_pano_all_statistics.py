import subprocess
import argparse
import sys
import os
from pathlib import Path

# 请根据你的实际情况修改这里的列表 (保持与 Unik3D 的测试集一致)
JSON_FILES = [
    "/workspace1/songcx/dataset/pvdepth/test_benchmark/setting_dynamic_fps20_len110.json",
    "/workspace1/songcx/dataset/pvdepth/test_benchmark/setting_dynamic_fps10_len90.json",
    "/workspace1/songcx/dataset/pvdepth/test_benchmark/setting_dynamic_fps02_len50.json",
    # ... 其他配置文件
]

SCRIPT_NAME = "infer_vipe_panorama_statistics.py"


def derive_profile_output_json(base_output_json, json_path):
    base_path = Path(base_output_json)
    json_stem = Path(json_path).stem
    if base_path.suffix:
        file_name = f"{base_path.stem}__{json_stem}{base_path.suffix}"
    else:
        file_name = f"{base_path.name}__{json_stem}.json"
    return str(base_path.with_name(file_name))

def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--json_path", help="Ignored")
    parser.add_argument("--profile_mode", default=None)
    parser.add_argument("--profile_clip_limit", type=int, default=None)
    parser.add_argument("--profile_output_json", default=None)
    known_args, passthrough = parser.parse_known_args()

    for i, json_path in enumerate(JSON_FILES):
        print(f"\n>>> [{i+1}/{len(JSON_FILES)}] Processing: {json_path}")
        
        if not os.path.exists(json_path):
            print(f"File not found: {json_path}")
            continue

        cmd = [sys.executable, SCRIPT_NAME, "--json_path", json_path] + passthrough

        if known_args.profile_mode is not None:
            cmd.extend(["--profile_mode", known_args.profile_mode])
        if known_args.profile_clip_limit is not None:
            cmd.extend(["--profile_clip_limit", str(known_args.profile_clip_limit)])
        if known_args.profile_output_json:
            derived_output_json = derive_profile_output_json(known_args.profile_output_json, json_path)
            cmd.extend(["--profile_output_json", derived_output_json])

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Error processing: {json_path}")
            # sys.exit(1) # 可选：出错停止

if __name__ == "__main__":
    main()
