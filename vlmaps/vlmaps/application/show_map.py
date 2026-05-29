# needed for debugging
import sys
sys.path.append("/home/user/code/vlmaps/vlmaps")

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





def askForFilter(classes):
    labels = classes[0,:]
    names = classes[1,:]
    print("Filter label:")
    inp = input(": ")
    try:
        label = int(inp)
    except:
        idx = np.where(names == inp)
        label = labels[idx]
        label = int(label.item())
    return label

def askForColor():
    print("Select color:")
    print("0: RGB")
    print("1: Semantic")
    print("2: Region")
    print("3: Instance")
    print("4: Counts")
    print("5: Hits")
    try:
        col = int(input(": "))
    except:
        col = 0
    return col

def askForType():
    print("Select map type:")
    print("0: Regular")
    print("1: Postprocessed")
    print("2: Instances")
    print("3: Predicted")
    print("4: VLMAP Instances")
    print("5: OUR Instances")
    print("6: Predicted postprocessed")
    try:
        inptype = int(input(": "))
        if(inptype == 1):
            mapType = MapType.POSTPROCESSED
        elif(inptype == 2):
            mapType = MapType.INSTANCES
        elif (inptype == 3):
            mapType = MapType.PREDICTED
        elif (inptype == 4):
            mapType = MapType.VLMAP_INSTANCES
        elif (inptype == 5):
            mapType = MapType.OUR_INSTANCES
        elif (inptype == 6):
            mapType = MapType.PREDICTED_POSTPROCESSED
        else:
            mapType = MapType.REGULAR
    except:
        mapType = MapType.REGULAR
    return mapType

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="show_map_cfg.yaml",
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
    inp = ""
    askType = config.ask_for_type
    cfgType = config.type
    askColors = config.ask_for_color
    col = config.color
    filter = config.filter
    askFilter = config.ask_for_filter
    filterLabel = config.filter_label
    encoder_name = config.map_config.visual_encoder.name
    voxel_map = config.voxel_map
    classes, labels, names, num_classes_orig = common.parseClassFile(config.classes, config.delimiter)

    if(askType):
        mapType = askForType()
    else:
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


    if(askColors):
        col = askForColor()


    # ███╗   ███╗ █████╗ ██╗███╗   ██╗    ██╗      ██████╗  ██████╗ ██████╗
    # ████╗ ████║██╔══██╗██║████╗  ██║    ██║     ██╔═══██╗██╔═══██╗██╔══██╗
    # ██╔████╔██║███████║██║██╔██╗ ██║    ██║     ██║   ██║██║   ██║██████╔╝
    # ██║╚██╔╝██║██╔══██║██║██║╚██╗██║    ██║     ██║   ██║██║   ██║██╔═══╝
    # ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║    ███████╗╚██████╔╝╚██████╔╝██║
    # ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝    ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝


    while inp != 'q':
        vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
        if(config.direct_path):
            print("direct path loading")
            vlmap.load_map_override(data_dirs[id], config.direct_path_path)
        else:
            vlmap.load_map(data_dirs[id], mapType)

        if(filter):
            if(askFilter):
                filterLabel = askForFilter(classes)

            print("filter with", filterLabel, type(filterLabel))
            idx = (vlmap.grid_semantic == filterLabel).squeeze()
            print("idx", idx.shape)
            vlmap.grid_region = vlmap.grid_region[idx]
            vlmap.grid_semantic = vlmap.grid_semantic[idx]
            vlmap.grid_instance = vlmap.grid_instance[idx]
            vlmap.grid_rgb = vlmap.grid_rgb[idx]
            vlmap.grid_feat = vlmap.grid_feat[idx]
            vlmap.grid_pos = vlmap.grid_pos[idx]

        if col == 1:
            colors = common.color_semantics(vlmap.grid_semantic)
            visualize_rgb_map_3d(vlmap.grid_pos, colors, voxel=voxel_map)
        elif col == 2:
            colors = common.color_regions(vlmap.grid_region)
            visualize_rgb_map_3d(vlmap.grid_pos, colors, voxel=voxel_map)
        elif col == 3:
            print("instance coloring")
            colors = common.color_instances(vlmap.grid_instance)
            visualize_rgb_map_3d(vlmap.grid_pos, colors, voxel=voxel_map)
        elif col == 4:
            choice = input("Density? y/n: ")
            if choice.lower() == "y":
                file = f"counts-map-density-{encoder_name}-{id}.npy"
            else:
                file = f"counts-map-{encoder_name}-{id}.npy"
            counts = np.load(file)
            colors = common.color_counts(counts)
            visualize_rgb_map_3d(vlmap.grid_pos, colors, voxel=voxel_map)
        elif col == 5:
            choice = input("Density? y/n: ")
            if choice.lower() == "y":
                file = f"hits-map-density-{encoder_name}-{id}.npy"
            else:
                file = f"hits-map-{encoder_name}-{id}.npy"
            hits = np.load(file)
            colors = common.color_counts(hits)
            visualize_rgb_map_3d(vlmap.grid_pos, colors, voxel=voxel_map)
        else:
            visualize_rgb_map_3d(vlmap.grid_pos, vlmap.grid_rgb, voxel=voxel_map)

        print("q: quit")
        if(askColors):
            print("d: change color type")
        if(askType):
            print("t: change map type")
        print("m: change map")

        if(col == 1):
            print("color: semantic")
        elif (col == 2):
            print("color: region")
        elif (col == 3):
            print("color: instance")
        elif (col == 4):
            print("color: counts")
        elif (col == 5):
            print("color: hits")
        else:
            print("color: rgb")

        inp = input(": ")
        if (inp == "q"):
            exit()
        elif(inp == "d"):
            if(askColors):
                col = askForColor()
        elif(inp == "t"):
            if(askType):
                mapType = askForType()
        elif(inp == "m"):
            print("0-9: new id")
            inp = input(": ")
            try:
                id = int(inp)
            except:
                print("Unrecognized input")
        else:
            continue


if __name__ == "__main__":
    main()
