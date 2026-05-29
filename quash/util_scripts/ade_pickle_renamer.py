import os
import re

def rename_pkl_files(base_path):
    # Regex pattern to match .pkl files without variation: "ADE_{split}_{num}.pkl"
    pattern_pkl_no_variation = re.compile(r"ADE_(train|val)_(\d+)\.pkl")

    # Traverse the directory and its subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".pkl"):
                match_pkl = pattern_pkl_no_variation.match(file)
                if match_pkl:
                    category = match_pkl.group(1)  # Either "train" or "validation"
                    num = match_pkl.group(2)  # The number in the filename

                    # Construct the new filename with the "ade_full" variation
                    new_filename = f"ADE_{category}_{num}_ade_full.pkl"

                    # Full paths
                    old_filepath = os.path.join(root, file)
                    new_filepath = os.path.join(root, new_filename)

                    # Rename the file
                    os.rename(old_filepath, new_filepath)
                    print(f"Renamed: {old_filepath} -> {new_filepath}")

if __name__ == "__main__":
    base_path = "/home/user/data/datasets/Datasets_HSN/ADE20K_2021_17_01/images"
    rename_pkl_files(base_path)
