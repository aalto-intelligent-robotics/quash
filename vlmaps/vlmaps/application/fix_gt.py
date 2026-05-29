# needed for debugging
import sys
sys.path.append("/home/user/code/vlmaps/vlmaps")

from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
from vlmaps.map.vlmap import MapType
import os
import numpy as np
import open3d as o3d
from mp_cmap import MATTERPORT_COLOR_MAP_160
import copy

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="fix_gt_cfg.yaml",
)
def main(config: DictConfig) -> None:
    outfile = f"{config.output}/{config.scene_id}_{config.map_config.cell_size}.data.npy"
    exists = os.path.isfile(outfile)
    if exists and not config.force:
        print("Parsed file already exists")
        exit(0)

    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    vlmap = VLMap(config.map_config, data_dir=data_dirs[config.scene_id])

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

    if(config.direct_path):
        print("direct path loading")
        vlmap.load_map_override(data_dirs[config.scene_id], config.direct_path_path)
    else:
        vlmap.load_map(data_dirs[config.scene_id], mapType)

    vlmap.grid_pos[:, 0] -= np.min(vlmap.grid_pos[:, 0])
    vlmap.grid_pos[:, 1] -= np.min(vlmap.grid_pos[:, 1])
    vlmap.grid_pos[:, 2] -= np.min(vlmap.grid_pos[:, 2])
    vlmaps_minx = np.min(vlmap.grid_pos[:, 0])
    vlmaps_miny = np.min(vlmap.grid_pos[:, 1])
    vlmaps_minz = np.min(vlmap.grid_pos[:, 2])
    vlmaps_maxx = np.max(vlmap.grid_pos[:, 0])
    vlmaps_maxy = np.max(vlmap.grid_pos[:, 1])
    vlmaps_maxz = np.max(vlmap.grid_pos[:, 2])
    print("VLMaps")
    print(f"x: {vlmaps_minx} - {vlmaps_maxx} | {vlmaps_maxx - vlmaps_minx}")
    print(f"y: {vlmaps_miny} - {vlmaps_maxy} | {vlmaps_maxy - vlmaps_miny}")
    print(f"z: {vlmaps_minz} - {vlmaps_maxz} | {vlmaps_maxz - vlmaps_minz}")
    vl_colors = get_colors(vlmap.grid_semantic.squeeze(1))
    # show(vlmap.grid_pos, vl_colors)
    # show(vlmap.grid_pos, vlmap.grid_rgb/255)
    vl_pcd = o3d.geometry.PointCloud()
    vl_pcd.points = o3d.utility.Vector3dVector(vlmap.grid_pos.astype(dtype=np.float32))

    path = "/home/user/hdd/3-openscene-maps/parsed-lseg-0.05/5LpN3gDmAk7_grid.data.npy"
    grid = np.load(path)
    semantic = np.load(path.replace("grid", "gt"))
    if "0.05" in path:
        mask = grid[2, :] < 40  # 0.05
    elif "0.1" in path:
        mask = grid[2, :] < 20  # 0.1
    elif "0.2" in path:
        mask = grid[2, :] < 10  # 0.2
    grid = grid[:, mask]
    semantic = semantic[mask]
    grid = np.transpose(grid)
    grid = grid[:, [1, 0, 2]]
    grid[:, 0] = -grid[:, 0]
    grid[:, 0] -= np.min(grid[:, 0])
    grid[:, 1] -= np.min(grid[:, 1])
    grid[:, 2] -= np.min(grid[:, 2])
    # grid[:, 2] -= 10
    os_colors = get_colors(semantic)

    openscene_minx = np.min(grid[:, 0])
    openscene_miny = np.min(grid[:, 1])
    openscene_minz = np.min(grid[:, 2])
    openscene_maxx = np.max(grid[:, 0])
    openscene_maxy = np.max(grid[:, 1])
    openscene_maxz = np.max(grid[:, 2])
    print("OpenScene")
    print(f"x: {openscene_minx} - {openscene_maxx} | {openscene_maxx - openscene_minx}")
    print(f"y: {openscene_miny} - {openscene_maxy} | {openscene_maxy - openscene_miny}")
    print(f"z: {openscene_minz} - {openscene_maxz} | {openscene_maxz - openscene_minz}")
    # show(grid, os_colors)
    # show_many([vlmap.grid_pos, grid], [vl_colors, os_colors])
    os_pcd = o3d.geometry.PointCloud()
    os_pcd.points = o3d.utility.Vector3dVector(grid.astype(dtype=np.float32))

    # sanity check - transform one cloud and find the transformation
    # rand = np.random.normal(0, 1, (3, 3))
    # Q, R = np.linalg.qr(rand)
    # t_rand = np.random.random(3)*10
    # T_rand = np.zeros((4, 4))
    # T_rand[0:3, 0:3] = Q
    # T_rand[0:3, 3] = t_rand
    # T_rand[3, 3] = 1
    # print(T_rand)
    # vl_pcd = copy.deepcopy(os_pcd).transform(T_rand)


    # rough transform
    source = o3d.geometry.PointCloud()
    mask = np.logical_and(vlmap.grid_pos[:, 0] > 200, vlmap.grid_pos[:, 1] > 200)
    source.points = o3d.utility.Vector3dVector(vlmap.grid_pos[mask].astype(dtype=np.float32))
    target = os_pcd
    threshold = 20

    trans_init = np.identity(4)
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    draw_registration_result(vl_pcd, os_pcd, trans_init)
    voxel_size = 2
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    draw_registration_result(vl_pcd, os_pcd, result_ransac.transformation)

    print("Transformation is:")
    print(result_ransac.transformation)
    # fine reg
    trans_init = result_ransac.transformation

    print("RANSAC alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 100000000, relative_fitness = 1e-5, relative_rmse = 1e-5))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(vl_pcd, os_pcd, reg_p2p.transformation)


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, confidence=0.999))
    return result

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def get_colors(semantic: np.ndarray) -> np.ndarray:
    colormap = MATTERPORT_COLOR_MAP_160
    colors = np.array([colormap.get(key, (0, 0, 0)) for key in semantic])
    colors = colors / 255
    return colors

def show(grid, colors = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
    _ = o3d.visualization.draw_geometries([voxel_grid])

def show_many(grids, colorss):
    voxel_grids = []

    for grid, colors in zip(grids, colorss):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(grid)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
        voxel_grids.append(voxel_grid)

    _ = o3d.visualization.draw_geometries(voxel_grids)

if __name__ == "__main__":
    main()
