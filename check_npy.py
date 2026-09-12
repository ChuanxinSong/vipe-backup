import numpy as np
import sys
import os

def check_npy(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        data = np.load(file_path)
        print(f"File: {file_path}")
        print(f"Dtype: {data.dtype}")
        print(f"Shape: {data.shape}")
        print(f"Min:   {np.nanmin(data)}")
        print(f"Max:   {np.nanmax(data)}")
        print(f"Mean:  {np.nanmean(data)}")
        print(f"Std:   {np.nanstd(data)}")
        
        # Check if there are any NaNs or Infs
        if np.isnan(data).any():
            print(f"Contains NaNs: Yes ({np.isnan(data).sum()} values)")
        else:
            print("Contains NaNs: No")
            
        if np.isinf(data).any():
            print(f"Contains Infs: Yes ({np.isinf(data).sum()} values)")
        else:
            print("Contains Infs: No")

    except Exception as e:
        print(f"Error loading {file_path}: {e}")

if __name__ == "__main__":
    path = "carla_benchmark_results/vipe_pano_pvdepth/town0210_1024_dynamic_fps20_len110/town02_path1_clip_0_seg0_distance.npy"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    check_npy(path)
