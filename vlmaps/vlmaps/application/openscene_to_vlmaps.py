import argparse
import numpy as np
from vlmaps.map.vlmap import VLMap
import hydra
from omegaconf import DictConfig
import os
from pathlib import Path
from vlmaps.utils.mapping_utils import save_3d_map


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_creation_cfg.yaml",
)
def main(config: DictConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=False, default=None)
    parser.add_argument("--output", "-o", type=str, required=False, default=None)
    args = parser.parse_args()
    in_path = args.input
    out_path = args.output

    if in_path is None:
        in_path = "/home/user/hdd/openscene-mp40/parsed-lseg-0.05/5LpN3gDmAk7_embedding.data.npy"
        out_path = "/home/user/Desktop/os-vlmap-test/5LpN3gDmAk7/vlmaps-openscene.h5df"

    dir = os.path.dirname(out_path)
    Path(dir).mkdir(exist_ok=True, parents=True)

    embedding = np.load(in_path)
    semantic = np.load(in_path.replace("semantic", "semantic"))
    grid = np.load(in_path.replace("semantic", "grid"))
    gt = np.load(in_path.replace("semantic", "gt"))
    instance = np.load(in_path.replace("semantic", "instance"))

    vlmap = VLMap(config.map_config, force=config.force)

    vlmap.grid_feat = embedding
    vlmap.grid_pos = grid
    vlmap.grid_semantic = semantic
    vlmap.grid_instance = instance
    vlmap.mapped_iter_list = 0
    vlmap.weight = np.zeros_like(semantic)
    vlmap.occupied_ids = np.zeros_like(semantic)
    vlmap.grid_rgb = np.zeros((semantic.shape[0], 3))
    vlmap.grid_region = np.zeros_like(semantic)

    save_3d_map(out_path, vlmap.grid_feat, vlmap.grid_pos, vlmap.weight, vlmap.occupied_ids, vlmap.mapped_iter_list, vlmap.grid_rgb, vlmap.grid_semantic, vlmap.grid_region, vlmap.grid_instance)

if __name__ == "__main__":
    main()