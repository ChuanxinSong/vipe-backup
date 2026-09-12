import os
import json
import random

def generate_video_list(directory, output_file, count=50):
    # Common video extensions
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv')
    
    all_video_paths = []
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return

    # List all files
    files = os.listdir(directory)
    
    for file in files:
        if file.lower().endswith(video_extensions):
            full_path = os.path.join(directory, file)
            all_video_paths.append(full_path)
    
    # Randomly sample 'count' videos
    if len(all_video_paths) > count:
        selected_paths = random.sample(all_video_paths, count)
    else:
        selected_paths = all_video_paths
        print(f"Warning: Only found {len(all_video_paths)} videos, which is less than {count}.")
    
    # Sort selected paths for better readability
    selected_paths.sort()
    
    # Write to JSON
    with open(output_file, 'w') as f:
        json.dump(selected_paths, f, indent=4)
    
    print(f"Successfully wrote {len(selected_paths)} random video paths to {output_file}")

if __name__ == "__main__":
    source_dir = "/data3/songcx/huggingface_cache/hub/datasets--genex-world--Genex-DB-World-Exploration/snapshots/47a59d2b0b9e29091934fc1a2aa6274a4ece1b43/Real-World"
    output_filename = "genex_realworld_top50.json"
    
    generate_video_list(source_dir, output_filename)
