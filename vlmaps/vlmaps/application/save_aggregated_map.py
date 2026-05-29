import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
from pathlib import Path
import numpy as np
from vlmaps.utils.mapping_utils import save_3d_map
import pickle


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="aggregate.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    maptype = config.type
    assert maptype, "Map type not set"
    assert maptype == "new" or maptype == "lora", "Unknown map type"

    encoder_name = config.map_config.visual_encoder.name.lower()
    assert encoder_name, "Encoder not set"

    path = f"{data_dirs[config.scene_id]}/vlmap"
    parsefolder = f"{path}/parsed-alldata-{encoder_name}"
    assert Path(f"{parsefolder}/predicted_embeddings_{maptype}.npy").exists(), "No predictions found"

    grid_feat = np.load(f"{parsefolder}/predicted_embeddings_{maptype}.npy")
    grid_pos = np.load(f"{parsefolder}/grid_pos.npy")
    grid_rgb = np.load(f"{parsefolder}/grid_rgb.npy")
    grid_semantic = np.load(f"{parsefolder}/grid_semantic.npy")
    grid_region = np.load(f"{parsefolder}/grid_region.npy")
    grid_instance = np.load(f"{parsefolder}/grid_instance.npy")
    occupied_ids = np.load(f"{parsefolder}/occupied_ids.npy")
    weight = np.load(f"{parsefolder}/weight.npy")
    if Path(f"{parsefolder}/mapped_iter_list.pickle").exists():
        with open(f"{parsefolder}/mapped_iter_list.pickle", "rb") as f:
            mapped_iter_list = pickle.load(f)
    else:
        mapped_iter_list = []

    save_3d_map(
        f"{path}/transformer-{maptype}-{encoder_name}.h5df",
        grid_feat,
        grid_pos,
        weight,
        occupied_ids,
        mapped_iter_list,
        grid_rgb,
        grid_semantic,
        grid_region,
        grid_instance,
    )
    print(f"Map {config.scene_id} - {maptype} created: {path}/transformer-{maptype}-{encoder_name}.h5df")

if __name__ == "__main__":
    main()
