import os
import argparse
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
from vlmaps.map.vlmap import VLMap
import hydra
from omegaconf import DictConfig
from pathlib import Path
from scipy.spatial.distance import cdist, euclidean
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from tqdm import tqdm
from vlmaps.utils.mapping_utils import save_3d_map

# ███╗   ███╗███████╗██████╗ ██╗ █████╗ ███╗   ██╗
# ████╗ ████║██╔════╝██╔══██╗██║██╔══██╗████╗  ██║
# ██╔████╔██║█████╗  ██║  ██║██║███████║██╔██╗ ██║
# ██║╚██╔╝██║██╔══╝  ██║  ██║██║██╔══██║██║╚██╗██║
# ██║ ╚═╝ ██║███████╗██████╔╝██║██║  ██║██║ ╚████║
# ╚═╝     ╚═╝╚══════╝╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def create_vector_median_map(vlmap: VLMap, load_path: str, save_path: str = "") -> None:
    for key, items in tqdm(vlmap.grid_histogram.items()):
        median = geometric_median(np.array(items))
        vlmap.grid_feat[key] = median
    vlmap.grid_histogram = None  # type: ignore

    load_file_name = Path(load_path).name
    load_folder = Path(load_path).parent
    if not save_path:
        root = Tk()
        root.withdraw()
        save_path = askdirectory(title="Select save folder", initialdir=load_folder)
        root.destroy()

    save_file_name = load_file_name.replace("alldata", "median")
    save_path = f"{save_path}/{save_file_name}"

    save_map(vlmap, save_path)


# ███████╗ █████╗ ███╗   ███╗██████╗ ██╗     ███████╗
# ██╔════╝██╔══██╗████╗ ████║██╔══██╗██║     ██╔════╝
# ███████╗███████║██╔████╔██║██████╔╝██║     █████╗
# ╚════██║██╔══██║██║╚██╔╝██║██╔═══╝ ██║     ██╔══╝
# ███████║██║  ██║██║ ╚═╝ ██║██║     ███████╗███████╗
# ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝


def representative_sample(vectors: np.ndarray) -> np.ndarray:
    mean = vectors.mean(axis=0)
    distance_matrix = cosine_similarity(vectors, mean.reshape(1, -1))
    best_fit_idx = np.argmax(distance_matrix)
    sample = vectors[best_fit_idx, :]
    return sample


def create_representative_map(vlmap: VLMap, load_path: str, save_path: str = "") -> None:
    for key, items in tqdm(vlmap.grid_histogram.items()):
        median = representative_sample(np.array(items))
        vlmap.grid_feat[key] = median
    vlmap.grid_histogram = None  # type: ignore

    load_file_name = Path(load_path).name
    load_folder = Path(load_path).parent
    if not save_path:
        root = Tk()
        root.withdraw()
        save_path = askdirectory(title="Select save folder", initialdir=load_folder)
        root.destroy()

    save_file_name = load_file_name.replace("alldata", "sample")
    save_path = f"{save_path}/{save_file_name}"

    save_map(vlmap, save_path)

# ███╗   ███╗ █████╗ ██╗███╗   ██╗    ███████╗██╗  ██╗███████╗
# ████╗ ████║██╔══██╗██║████╗  ██║    ██╔════╝╚██╗██╔╝██╔════╝
# ██╔████╔██║███████║██║██╔██╗ ██║    █████╗   ╚███╔╝ ███████╗
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║    ██╔══╝   ██╔██╗ ╚════██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║    ██║     ██╔╝ ██╗███████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝    ╚═╝     ╚═╝  ╚═╝╚══════╝


def save_map(vlmap: VLMap, save_path: str):
    save_3d_map(
        save_path,
        vlmap.grid_feat,
        vlmap.grid_pos,
        vlmap.weight,
        vlmap.occupied_ids,
        vlmap.mapped_iter_list,
        vlmap.grid_rgb,
        vlmap.grid_semantic,
        vlmap.grid_region,
        vlmap.grid_instance,
        vlmap.grid_histogram,
    )


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="median.yaml",
)
def main(config: DictConfig):
    path = config.dir
    id = config.id
    type = config.type
    save_path = config.save_path

    if path is None:
        root = Tk()
        root.withdraw()
        path = askopenfilename(title='Select Map', initialdir="/home/user/hdd/datasets/vlmaps_dataset/5LpN3gDmAk7_1/vlmap")
        root.destroy()

    assert "alldata" in path, "Must be alldata map!"

    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    vlmap.load_map_override(data_dirs[id], path)

    if type > 0:
        choice = str(type)
    else:
        print("Create:")
        print("1: Vector median map")
        print("2: Representative sample map")
        choice = input(": ")
    if choice == "1":
        create_vector_median_map(vlmap, path, save_path)
    elif choice == "2":
        create_representative_map(vlmap, path, save_path)
    else:
        print("Unknown choice")
    print("Done")

if __name__ == "__main__":
    main()
