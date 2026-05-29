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
from embeddings.embeddingCreator import EmbeddingCreator
from vlmaps.utils.mapping_utils import save_3d_map
import utils.common as common
import os

def distance(a, b):
    div = (np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2))
    if (div == 0):
        div = 1e-6
    d = np.dot(a, b)/div
    if (d > 1):
        d = 1
    if (d < -1):
        d = -1
    return d

def classifyMap(embeddings, queries, labels):
    # get the predicted embeddings
    distances = np.full(
        (len(queries), embeddings.shape[0]), -1, dtype=np.float32)

    for i in tqdm(range(len(queries)), desc="Classification", leave=False):
        query = queries[i]
        for j in tqdm(range(embeddings.shape[0]), disable=True, desc="Distances", leave=False):
            e = embeddings[j, :]
            d = distance(e, query)
            distances[i,j] = d

    # find the argmax of the embeddings (label with highest value)
    maxes = np.argmax(distances, axis=0)
    maxLabels = labels[maxes]

    return maxLabels


# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_classification.yaml",
)
def main(config: DictConfig) -> None:
    batch = config.batch

    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    id = config.scene_id

    force = config.force
    save_path = Path(data_dirs[id]) / "vlmap" / f"{config.map_config.map_prefix}-predicted-{config.map_config.cell_size}.h5df"
    if os.path.isfile(save_path):
        if not force:
            print(f"Map exists, exiting: {save_path}")
            exit(0)
        else:
            print("force = True, overwriting exiting map")
            os.remove(save_path)

    vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    vlmap.load_map(data_dirs[id])

    creator = EmbeddingCreator("cuda", model_name=config.map_config.visual_encoder.vlm_version)

    classes, labels, names, num_classes_orig = common.parseClassFile(config.classes, config.delimiter)

    queries = []
    for i in range(len(names)):
        queries.append(creator.get_text_embedding(names[i]))

    maxLabels = classifyMap(vlmap.grid_feat, queries, labels)

    # convert
    maxLabels = maxLabels.reshape(-1, 1)
    maxLabels = maxLabels.astype(dtype=np.uint8)

    # print(vlmap.grid_semantic.dtype, vlmap.grid_semantic.shape)
    # print(maxLabels.dtype, maxLabels.shape)

    # replace semantics
    vlmap.grid_semantic = maxLabels

    if (not batch):
        #! visualize results
        semantic_colors = common.color_semantics(vlmap.grid_semantic)
        visualize_rgb_map_3d(vlmap.grid_pos, semantic_colors)
        inp = input("save y/n? ")
    else:
        inp = "y"

    if (inp == 'y'):
        print("saving...")
        save_path = Path(data_dirs[id]) / "vlmap" / f"{config.map_config.map_prefix}-predicted-{config.map_config.cell_size}.h5df"
        save_3d_map(save_path, vlmap.grid_feat, vlmap.grid_pos, vlmap.weight, vlmap.occupied_ids,
                    vlmap.mapped_iter_list, vlmap.grid_rgb, vlmap.grid_semantic, vlmap.grid_region)
        print("saved")
    else:
        print("not saving")
    exit()


if __name__ == "__main__":
    main()
