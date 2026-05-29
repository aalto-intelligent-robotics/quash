import os
import argparse
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
import open3d as o3d
import utils.common as common
from vlmaps.map.vlmap import VLMap
import hydra
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm

def visualize_rgb_map_3d(pc: np.ndarray, rgb: np.ndarray, name: str = "Default name"):
    grid_rgb = rgb / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(grid_rgb)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
    _ = o3d.visualization.draw_geometries([voxel_grid], window_name=name)

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="show_map_cfg.yaml",
)
def main(config: DictConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, required=False)
    parser.add_argument("-i", "--id", type=int, required=False, default=0)
    args = parser.parse_args()
    path = args.dir
    id = args.id

    if not path:
        root = Tk()
        root.withdraw()
        path = askopenfilename(title='Select Map', initialdir="/home/user/hdd/datasets/vlmaps_dataset/5LpN3gDmAk7_1/vlmap-alldata")
        root.destroy()

    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    vlmap.load_map_override(data_dirs[id], path)

    variances = None
    total_variances = None
    while True:
        print("coloring:")
        print("1: variance")
        print("2: variance (log)")
        print("3: total variance")
        print("4: total variance (log)")
        choice = input(": ")
        color = None
        if choice == "q":
            exit()
        elif choice == "1" or choice == "2":
            log = choice == "2"
            exp = 0.5
            if log:
                try:
                    exp = float(input("log scaling: "))
                except:
                    exp = 0.5
            if variances is None:
                variances = np.zeros(vlmap.grid_semantic.shape[0], dtype=np.float32)
                for key, items in tqdm(vlmap.grid_histogram.items()):
                    array = np.array(items)
                    var = np.mean(np.var(array, axis=0))
                    variances[key] = var
            color = common.color_counts(variances, normalize=True, log=log, log_exp=exp)
        elif choice == "3" or choice == "4":
            log = choice == "4"
            exp = 1.0
            if log:
                try:
                    exp = float(input("log scaling: "))
                except:
                    exp = 0.5
            if total_variances is None:
                total_variances = np.zeros(vlmap.grid_semantic.shape[0], dtype=np.float32)
                for key, items in tqdm(vlmap.grid_histogram.items()):
                    if len(items) < 2:
                        var = 0
                    else:
                        array = np.array(items)
                        var = np.trace(np.cov(array, rowvar=False)) / array.shape[1]
                    total_variances[key] = var
            color = common.color_counts(total_variances, normalize=True, log=log, log_exp=exp)
        else:
            print("Unknown selection")

        if color is None:
            continue

        visualize_rgb_map_3d(vlmap.grid_pos, color)



if __name__ == "__main__":
    main()