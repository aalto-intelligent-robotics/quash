from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_3d,
    visualize_rgb_map_3d_instances,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_heatmap_3d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)
import numpy as np
from tqdm import tqdm
from vlmaps.utils.mapping_utils import save_3d_map
import utils.common as common
from vlmaps.map.vlmap import MapType
import os

class Region:
    def __init__(self, node, label, size, neighbors):
        self.nodes = []
        self.nodes.append(node)
        self.label = label
        self.size = size
        self.neighbors = neighbors


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="vlmap_instance_segmentation.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    batch = config.batch
    id = config.scene_id

    force = config.force
    save_path = Path(data_dirs[id]) / "vlmap" / f"{config.map_config.map_prefix}-vlmaps-instances-{config.map_config.cell_size}.h5df"
    if os.path.isfile(save_path):
        if not force:
            print(f"Map exists, exiting {save_path}")
            exit(0)
        else:
            print("force = True, overwriting exiting map")
            os.remove(save_path)

    vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    vlmap.load_map(data_dirs[id], MapType.PREDICTED)

    classes, labels, names, num_classes_orig = common.parseClassFile(config.classes, config.delimiter)
    vlmap._init_clip(config.map_config.visual_encoder.vlm_version)
    vlmap.init_categories(mp3dcat[1:-1])
    _ = vlmap.generate_obstacle_map()
    #_ = vlmap.generate_cropped_obstacle_map(obstacle_map)

    vlmap.grid_instance = np.full_like(vlmap.grid_semantic, 0, dtype=np.int32)

    # labels = [3]
    # names = ["chair"]

    instance_idx = 0

    for i in tqdm(range(len(labels)), leave=False):
        # print("*"*80)
        name = names[i]
        label = labels[i]
        # print("label:", name, label)
        try:
            contours, centers, bbox_list, mask = vlmap.get_pos(name)
            if(contours is None or len(contours) == 0):
                # print("no instances")
                continue

            minx = np.min(vlmap.grid_pos[:, 0])
            miny = np.min(vlmap.grid_pos[:, 1])

            # masked area
            masked = []
            masked_col = []

            for i in tqdm(range(vlmap.grid_pos.shape[0]), leave=False):
                xyz = vlmap.grid_pos[i]
                xi = xyz[0]-minx
                yi = xyz[1]-miny
                if(mask[xi,yi]):
                    masked.append(xyz)
                    masked_col.append(vlmap.grid_rgb[i])
                    vlmap.grid_semantic[i] = label

            masked = np.array(masked)
            masked_col = np.array(masked_col)
        except Exception as e:
            # print(e)
            continue

        if(not batch):
            visualize_rgb_map_3d(np.array(masked), np.array(masked_col))

    # print(vlmap.grid_semantic.shape)
    if(not batch):
        #! visualize results
        semantic_colors = common.color_semantics(vlmap.grid_semantic)
        visualize_rgb_map_3d(vlmap.grid_pos, semantic_colors)
        inp = input("save y/n? ")
    else:
        inp = "y"

    if(inp == 'y'):
        print("saving...")
        save_path = Path(data_dirs[id]) / "vlmap" / f"{config.map_config.map_prefix}-vlmaps-instances-{config.map_config.cell_size}.h5df"
        save_3d_map(save_path, vlmap.grid_feat, vlmap.grid_pos, vlmap.weight, vlmap.occupied_ids, vlmap.mapped_iter_list, vlmap.grid_rgb, vlmap.grid_semantic, vlmap.grid_region)
        print("saved")
    else:
        print("not saving")

        print("contours:", len(contours))
        print("centers:", len(centers))
        print("bbox_list:", len(bbox_list))

if __name__ == "__main__":
    main()
