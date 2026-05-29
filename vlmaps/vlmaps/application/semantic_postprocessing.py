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
import os

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
    config_name="map_postprocessing.yaml",
)
def main(config: DictConfig) -> None:
    id = config.scene_id
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    force = config.force

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

    # check if exists
    if(mapType == MapType.PREDICTED):
        save_path = Path(data_dirs[id]) / "vlmap" / f"{config.map_config.map_prefix}-predicted-postprocessed-{config.map_config.cell_size}.h5df"
    else:
        save_path = Path(data_dirs[id]) / "vlmap" / f"{config.map_config.map_prefix}-postprocessed-{config.map_config.cell_size}.h5df"

    if os.path.isfile(save_path):
        if not force:
            print(f"Map exists, exiting {save_path}")
            exit(0)
        else:
            print("force = True, overwriting exiting map")
            os.remove(save_path)

    # nominal
    vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    vlmap.load_map(data_dirs[id], mapType)
    batch = config.batch

    #! settings
    iterations = 20
    min_num_neighbors = 5
    find_neighbors = 1
    majority_neighbors = 2

    #! visualize intial state
    # semantic_colors = common.color_semantics(vlmap.grid_semantic)
    # visualize_rgb_map_3d(vlmap.grid_pos, semantic_colors)

    #! postprocess
    minx = np.min(vlmap.grid_pos[:,0])
    miny = np.min(vlmap.grid_pos[:,1])
    minz = np.min(vlmap.grid_pos[:,2])
    vlmap.grid_pos[:, 0] = vlmap.grid_pos[:, 0] - minx
    vlmap.grid_pos[:, 1] = vlmap.grid_pos[:, 1] - miny
    vlmap.grid_pos[:, 2] = vlmap.grid_pos[:, 2] - minz
    maxx = np.max(vlmap.grid_pos[:,0])
    maxy = np.max(vlmap.grid_pos[:,1])
    maxz = np.max(vlmap.grid_pos[:,2])

    dx = np.diff(np.sort(vlmap.grid_pos[:, 0]))
    mindx = dx[dx > 0].min()
    dy = np.diff(np.sort(vlmap.grid_pos[:, 1]))
    mindy = dy[dy > 0].min()
    dz = np.diff(np.sort(vlmap.grid_pos[:, 2]))
    mindz = dz[dz > 0].min()

    # copy to 3d grid
    grid = np.full((maxx+1, maxy+1, maxz+1), -1)
    occupied = np.full((maxx+1, maxy+1, maxz+1), False)
    corrected = np.full((maxx+1, maxy+1, maxz+1), False)
    indices = np.full((maxx+1, maxy+1, maxz+1), -1)
    for i in tqdm(range(vlmap.grid_pos.shape[0])):
        xyz = vlmap.grid_pos[i]
        grid[xyz[0], xyz[1], xyz[2]] = vlmap.grid_semantic[i]
        occupied[xyz[0], xyz[1], xyz[2]] = True
        indices[xyz[0], xyz[1], xyz[2]] = i

    # fix cells
    for i in range(iterations):
        num_corrected = 0
        for x in range(0, maxx):
            for y in range(0, maxy):
                for z in range(0, maxz):
                    oc = occupied[x,y,z]
                    cor = corrected[x,y,z]
                    l = grid[x,y,z]
                    if(oc and not cor):
                        n = get_neighbors(grid, x, y, z, find_neighbors)
                        if(np.any(n)):
                            marchingLabels = np.sum(n == l)
                            if(marchingLabels < min_num_neighbors):
                                # take more neighbors for majority label
                                n = get_neighbors(grid, x, y, z, majority_neighbors)
                                counts = np.bincount(n)
                                majority = np.argmax(counts)
                                grid[x,y,z] = majority
                                corrected[x,y,z] = True
                                num_corrected += 1

        print("Iteration", (i+1), "/", iterations,
              ": Corrected", num_corrected, "cells")
        if(num_corrected == 0):
            break

    for x in range(0, maxx):
        for y in range(0, maxy):
            for z in range(0, maxz):
                idx = indices[x, y, z]
                if(idx >= 0):
                    l = grid[x, y, z]
                    vlmap.grid_semantic[idx] = l

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
        semantic_colors = common.color_semantics(vlmap.grid_semantic)
        visualize_rgb_map_3d(vlmap.grid_pos, semantic_colors)
        inp = input("save y/n? ")
    else:
        inp = "y"

    if(inp == 'y'):
        print("saving...")
        if(mapType == MapType.PREDICTED):
            save_path = Path(data_dirs[id]) / "vlmap" / f"{config.map_config.map_prefix}-predicted-postprocessed-{config.map_config.cell_size}.h5df"
        else:
            save_path = Path(data_dirs[id]) / "vlmap" / f"{config.map_config.map_prefix}-postprocessed-{config.map_config.cell_size}.h5df"
        save_3d_map(save_path, vlmap.grid_feat, vlmap.grid_pos, vlmap.weight, vlmap.occupied_ids, vlmap.mapped_iter_list, vlmap.grid_rgb, vlmap.grid_semantic, vlmap.grid_region)
        print("saved")
    else:
        print("not saving")
    exit()


if __name__ == "__main__":
    main()
