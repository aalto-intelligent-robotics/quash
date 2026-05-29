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


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    id = config.scene_id
    vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    vlmap.load_map(data_dirs[id])

    vlmap._init_clip()
    vlmap.init_categories(mp3dcat[1:-1])

    query = "chair"
    obstacle_map = vlmap.generate_obstacle_map()
    cropped_obstacle_map = vlmap.generate_cropped_obstacle_map(obstacle_map)
    contours, centers, bbox_list = vlmap.get_pos(query)

    visualize_rgb_map_3d_instances(vlmap.grid_pos, vlmap.grid_rgb, contours, centers, bbox_list)

    # mask = vlmap.index_map(query, with_init_cat=True)
    # visualize_masked_map_3d(vlmap.grid_pos, mask, vlmap.grid_rgb)

    print("*"*80)
    print("contours")
    print(contours)
    print("*"*80)
    print("centers")
    print(centers)
    print("*"*80)
    print("bbox_list")
    print(bbox_list)

    print("*"*80)
    print("contours:", len(contours))
    print("centers:", len(centers))
    print("bbox_list:", len(bbox_list))


if __name__ == "__main__":
    main()
