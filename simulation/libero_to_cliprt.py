import os
import h5py
import json
import numpy as np
import glob
import cv2
import argparse
from tqdm import tqdm
from PIL import Image

def process_libero_to_cliprt(hdf5_file, output_base_dir):
    """HDF5 파일을 변환하여 개별 timestep JSON 데이터 생성"""
    file_name = os.path.basename(hdf5_file).replace("_demo.hdf5", "")
    instruction = file_name.replace("_", " ")
    output_dir = os.path.join(output_base_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(hdf5_file, "r") as f:
        if "data" not in f:
            print(f"❌ Error: 'data' group doesn't exist. ({hdf5_file})")
            return
        
        data_group = f["data"]
        for demo in tqdm(sorted(data_group.keys()), desc=f"Processing {file_name}", leave=False):
            demo_data = data_group[demo]
            actions = np.array(demo_data["actions"])
            joint_states = np.array(demo_data["obs/joint_states"])
            robot_states = np.array(demo_data["robot_states"])
            agentview_rgb = np.array(demo_data["obs/agentview_rgb"])

            demo_output_dir = os.path.join(output_dir, demo)
            os.makedirs(demo_output_dir, exist_ok=True)
            
            for i in tqdm(range(len(actions)), desc=f"Processing {demo}", leave=False):
                image_filename = f"{demo}_timestep_{i:04d}.png"
                image_path = os.path.join(demo_output_dir, image_filename)
                
                flipped_image = cv2.flip(agentview_rgb[i], 0)  # OpenCV를 사용한 상하 반전
                Image.fromarray(flipped_image).save(image_path)

                timestep_data = {
                    "joint": joint_states[i].tolist(),
                    "instruction": instruction,
                    "supervision": "",  # Supervision 추가 가능
                    "action": actions[i].tolist(),
                    "image_path": image_filename,
                    "states": robot_states[i].tolist(),
                }
                
                json_path = os.path.join(demo_output_dir, f"{demo}_timestep_{i:04d}.json")
                with open(json_path, "w") as json_file:
                    json.dump(timestep_data, json_file, indent=4)
    
        print(f"✅ {file_name}: All timestep data saved successfully!")

def process_all_hdf5(input_dir, output_base_dir):
    """지정된 디렉토리 내 모든 HDF5 파일을 변환"""
    hdf5_files = glob.glob(os.path.join(input_dir, "*.hdf5"))
    print(f"📂 Found {len(hdf5_files)} files. Processing...")
    for hdf5_file in tqdm(hdf5_files, desc="Processing HDF5 Files"):
        process_libero_to_cliprt(hdf5_file, output_base_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Libero HDF5 to CLIP-RT Format")
    parser.add_argument("-i", "--input", required=True, help="Dataset name (e.g., 'goal' -> 'libero_goal')")
    args = parser.parse_args()
    
    input_directory = f"libero_{args.input}"
    output_directory = f"libero_{args.input}_individual_demo"
    
    print(f"🔹 Input Directory: {input_directory}")
    print(f"🔹 Output Directory: {output_directory}")
    
    process_all_hdf5(input_directory, output_directory)
