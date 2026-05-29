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
import argparse
from vlmaps.map.vlmap import MapType
import os

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_parsing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    Path(config.output).mkdir(parents=True, exist_ok=True)
    outfile = f"{config.output}/{config.scene_id}.data.npy"
    exists = os.path.isfile(outfile)
    if exists and not config.force:
        print("Parsed file already exists")
        exit(0)

    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    vlmap = VLMap(config.map_config, data_dir=data_dirs[config.scene_id])

    cfgType = config.type
    if(cfgType == 1):
        mapType = MapType.POSTPROCESSED
    elif(cfgType == 2):
        mapType = MapType.INSTANCES
    elif (cfgType == 3):
        mapType = MapType.PREDICTED
    elif (cfgType == 4):
        mapType = MapType.VLMAP_INSTANCES
    elif (cfgType == 5):
        mapType = MapType.OUR_INSTANCES
    elif (cfgType == 6):
        mapType = MapType.PREDICTED_POSTPROCESSED
    else:
        mapType = MapType.REGULAR

    vlmap.load_map(data_dirs[config.scene_id], mapType)

    regions = config.regions

    len = vlmap.grid_feat.shape[0]
    data  = np.zeros((len, 4, config.map_config.visual_encoder.embedding_size))
    #data = np.concatenate((vlmap.grid_feat, vlmap.grid_semantic), axis=1)
    data[:, 0, :] = vlmap.grid_semantic
    data[:, 1, :] = vlmap.grid_instance
    data[:, 2, :] = vlmap.grid_region
    data[:, 3, :] = vlmap.grid_feat


    np.save(outfile, data)
    #print(np.unique(data[:,1,0]))
    print("parsing complete")


if __name__ == "__main__":
    main()
