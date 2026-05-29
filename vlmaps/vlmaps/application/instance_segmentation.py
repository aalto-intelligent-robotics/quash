from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
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
import numpy as np
from vlmaps.utils.mapping_utils import save_3d_map
from tqdm import tqdm
from vlmaps.map.vlmap import MapType
import utils.common as common
from utils.instance_segmentation import instance_segmentation
import os

class Region:
    def __init__(self, node, label, size, neighbors):
        self.nodes = []
        self.nodes.append(node)
        self.label = label
        self.size = size
        self.neighbors = neighbors


def get_neighbors(grid, x, y, z, numNeighbors):
    neighbors = []
    for dx in range(-1*numNeighbors, numNeighbors + 1):
        for dy in range(-1*numNeighbors, numNeighbors + 1):
            for dz in range(-1*numNeighbors, numNeighbors + 1):
                # indentity, not neighbor
                if(dx == 0 and dy == 0 and dz == 0):
                    continue
                # check bounds
                if(grid.shape[0] > x+dx and x+dx >= 0):
                    if(grid.shape[1] > y+dy and y+dy >= 0):
                        if(grid.shape[2] > z+dz and z+dz >= 0):
                            label = grid[x + dx, y + dy, z + dz]
                            if(label >= 0):
                                neighbors.append(label)
    return np.array(neighbors)

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="instance_segmentation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    id = config.scene_id
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    force = config.force

    cfgType = config.type
    if (cfgType == 1):
        mapType = MapType.POSTPROCESSED
    elif (cfgType == 2):
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

    if(mapType == MapType.PREDICTED or mapType == MapType.PREDICTED_POSTPROCESSED):
        save_path = Path(data_dirs[id]) / "vlmap" / f"{config.map_config.map_prefix}-our-instances-{config.map_config.cell_size}.h5df"
    else:
        save_path = Path(data_dirs[id]) / "vlmap" / f"{config.map_config.map_prefix}-instances-{config.map_config.cell_size}.h5df"

    if os.path.isfile(save_path):
        if not force:
            print(f"Map exists, exiting: {save_path}")
            exit(0)
        else:
            print("force = True, overwriting exiting map")
            os.remove(save_path)

    print(f"create new map {save_path}")

    vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    map_loaded = vlmap.load_map(data_dirs[id], mapType)
    if not map_loaded:
        exit(0)

    batch = config.batch
    #! important
    instanceMinSize = config.instance_min_size

    print("Instance segmentation")
    insta = len(np.unique(vlmap.grid_instance))
    print("Map has", insta, "instances")

    gt = vlmap.grid_semantic.reshape(-1)
    instances = instance_segmentation(gt, vlmap.grid_pos, instanceMinSize, 5, 5)
    vlmap.grid_instance = instances.reshape(-1, 1)

    insta = len(np.unique(vlmap.grid_instance))
    print("Map has now", insta, "instances")

    # find neighbors, and with that, anomalies (single voxels surrounded by other labels)
    # fix
    # move back to array
    # save

    # idx = np.argsort(vlmap.grid_pos, axis=0)
    # vlmap.grid_pos = np.take_along_axis(vlmap.grid_pos, idx, axis=0)
    # vlmap.grid_semantic = np.take_along_axis(vlmap.grid_semantic, idx[:,0], axis=0)
    # #for i in range(len(vlmap.grid_pos)):
    # for i in range(5):
    #     xyz = vlmap.grid_pos[i]
    #     l = vlmap.grid_semantic[i]
    #     print(xyz, " ", l)

    if(not batch):
        #! visualize results
        semantic_colors = common.color_instances(vlmap.grid_instance)
        visualize_rgb_map_3d(vlmap.grid_pos, semantic_colors)

        inp = input("save y/n? ")
    else:
        inp = "y"

    if(inp == 'y'):
        print("saving...")
        if(mapType == MapType.PREDICTED or mapType == MapType.PREDICTED_POSTPROCESSED):
            save_path = Path(data_dirs[id]) / "vlmap" / f"{config.map_config.map_prefix}-our-instances-{config.map_config.cell_size}.h5df"
        else:
            save_path = Path(data_dirs[id]) / "vlmap" / f"{config.map_config.map_prefix}-instances-{config.map_config.cell_size}.h5df"
        save_3d_map(save_path, vlmap.grid_feat, vlmap.grid_pos, vlmap.weight, vlmap.occupied_ids, vlmap.mapped_iter_list, vlmap.grid_rgb, vlmap.grid_semantic, vlmap.grid_region, vlmap.grid_instance)
        print("saved")
    else:
        print("not saving")
    exit()


if __name__ == "__main__":
    main()
