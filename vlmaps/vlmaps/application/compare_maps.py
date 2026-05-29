from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
import numpy as np
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_3d,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_heatmap_3d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)
from vlmaps.map.vlmap import MapType
import utils.common as common
import os



@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="compare_maps_cfg.yaml",
)

# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝


def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    id = config.scene_id
    gundam_map = VLMap(config.map_config, data_dir=data_dirs[id])
    gundam_map.load_map_override(data_dirs[id], os.environ['DATA_DIR'] + "/vlmaps_dataset/5LpN3gDmAk7_1/vlmap/vlmaps-remade.h5df")

    own_map = VLMap(config.map_config, data_dir=data_dirs[id])
    own_map.load_map_override(data_dirs[id], os.environ['DATA_DIR'] + "/vlmaps_dataset/5LpN3gDmAk7_1/vlmap/vlmaps-old2.h5df")

    print("g:", gundam_map.grid_semantic.shape)
    print("o:", own_map.grid_semantic.shape)



if __name__ == "__main__":
    main()
