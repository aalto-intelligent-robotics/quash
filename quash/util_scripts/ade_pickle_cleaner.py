import os
import re

def rename_pkl_files(base_path):
    # Regex pattern to match .pkl files without variation: "ADE_{split}_{num}.pkl"
    pattern_pkl = re.compile(r"ADE_(train|val)_(\d+)_(ade_all)\.pkl")

    # Traverse the directory and its subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".pkl"):
                match_pkl = pattern_pkl.match(file)
                if match_pkl:
                    # Full paths
                    path = os.path.join(root, file)

                    # Rename the file
                    os.remove(path)
                    print(f"Removed: {path}")

if __name__ == "__main__":
    base_path = "/home/user/data/datasets/Datasets_HSN/ADE20K_2021_17_01/images"
    rename_pkl_files(base_path)
