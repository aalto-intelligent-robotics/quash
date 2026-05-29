from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
import os

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_creation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    if config.store_all and not config.force_store_all:
        print("You are about to create a very large map with all the measurements. Are you sure?")
        choice = input("y/n: ")
        if choice != "y":
            exit()

    vlmap = VLMap(config.map_config, force=config.force, store_all=config.store_all)
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    vlmap.create_map(data_dirs[config.scene_id])


if __name__ == "__main__":
    main()
