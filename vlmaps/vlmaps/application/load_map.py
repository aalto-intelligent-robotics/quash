from vlmaps.map.vlmap import VLMap
import hydra
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
import pickle

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="load_map_cfg.yaml",
)
def main(config: DictConfig):
    print("load map")
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    id = config.scene_id
    vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    vlmap.load_map_override(data_dirs[id], config.direct_path_path)
    encoder_name = config.map_config.visual_encoder.name.lower()

    print("read map from:", vlmap.map_save_path)
    print("len(vlmap.mapped_iter_list):", len(vlmap.mapped_iter_list))
    print("vlmap.grid_feat.shape:", vlmap.grid_feat.shape)
    print("vlmap.grid_pos.shape:", vlmap.grid_pos.shape)
    print("vlmap.grid_rgb.shape:", vlmap.grid_rgb.shape)
    if vlmap.grid_semantic is not None:
        print("vlmap.grid_semantic.shape:", vlmap.grid_semantic.shape)
    if vlmap.grid_region is not None:
        print("vlmap.grid_region.shape:", vlmap.grid_region.shape)
    if vlmap.grid_instance is not None:
        print("vlmap.grid_instance.shape:", vlmap.grid_instance.shape)
    print("vlmap.occupied_ids.shape:", vlmap.occupied_ids.shape)
    print("vlmap.weight.shape:", vlmap.weight.shape)

    num_keys = 0
    sum_vals = 0
    for key, value in vlmap.grid_histogram.items():
        num_keys += 1
        sum_vals += len(value)
    avg_vals = sum_vals / num_keys
    print(f"vlmap.grid_histogram: {num_keys} keys w/ avg {avg_vals} vals")
    print(" ")

    parse = config.parse
    if not parse:
        choice = input("Parse? y/n: ")
        if choice.lower() == "y":
            parse = True
    if parse:
        path = Path(config.direct_path_path)
        folder = path.parent
        parsefolder = f"{folder}/parsed-alldata-{encoder_name}/"
        Path(parsefolder).mkdir(exist_ok=True, parents=True)

        np.save(f"{parsefolder}/grid_feat", vlmap.grid_feat)
        np.save(f"{parsefolder}/grid_pos", vlmap.grid_pos)
        np.save(f"{parsefolder}/grid_rgb", vlmap.grid_rgb)
        np.save(f"{parsefolder}/grid_semantic", vlmap.grid_semantic)
        np.save(f"{parsefolder}/grid_region", vlmap.grid_region)
        np.save(f"{parsefolder}/grid_instance", vlmap.grid_instance)
        np.save(f"{parsefolder}/occupied_ids", vlmap.occupied_ids)
        np.save(f"{parsefolder}/weight", vlmap.weight)
        if vlmap.mapped_iter_list is not None:
            with open(f"{parsefolder}/mapped_iter_list.pickle", "wb") as f:
                pickle.dump(vlmap.mapped_iter_list, f)
        with open(f"{parsefolder}/grid_histogram.pickle", "wb") as f:
            pickle.dump(vlmap.grid_histogram, f)


if __name__ == "__main__":
    main()
