from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
import numpy as np
from vlmaps.map.vlmap import MapType
import tkinter.filedialog
from tqdm import tqdm
from typing import List, Tuple, Optional, Any
import tkinter
import tkinter.filedialog

class Dialog():
    def __init__(self):
        pass

    def __del__(self):
        pass

    def __enter__(self):
        self.root = tkinter.Tk()
        self.root.withdraw()

    def __exit__(self, exc_type, exc_value, traceback):
        self.root.destroy()

def validate_path(input: Any) -> str:
    if isinstance(input, str):
        return input
    else:
        exit()

def asksaveasfilename(title: str, initialdir: str) -> str:
    with Dialog():
        path = tkinter.filedialog.asksaveasfilename(title=title, initialdir=initialdir)
    return validate_path(path)

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="parameter_tuning.yaml",
)
def main(config: DictConfig) -> None:
    ids = list(range(10))
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([str(x) for x in data_dir.iterdir() if x.is_dir()])

    semantic = []
    features = []

    for id in tqdm(ids):
        maptype = MapType.REGULAR
        # GT
        gt = VLMap(config.map_config, data_dir=data_dirs[id])
        gt_success = gt.load_map(data_dirs[id], maptype)
        assert gt_success, "Map loading failed"

        semantic.append(gt.grid_semantic)
        features.append(gt.grid_feat)

    semantic_arr = np.concatenate(semantic)
    features_arr = np.concatenate(features)
    np.save(asksaveasfilename("Save as", "."), semantic_arr)
    np.save(asksaveasfilename("Save as", "."), features_arr)


if __name__ == "__main__":
    main()
