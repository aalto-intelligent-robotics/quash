import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
import math


def visualize_rgb_map_3d(pc: np.ndarray, rgb: np.ndarray, name: str = "Default name", voxel: bool = True):
    grid_rgb = rgb / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(grid_rgb)

    if voxel:
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
        _ = o3d.visualization.draw_geometries([voxel_grid], window_name=name)
    else:
        _ = o3d.visualization.draw_geometries([pcd], window_name=name)

    # # vis = o3d.Visualization.Visualizer()
    # ctr = window.get_view_control()
    # parameters = o3d.io.read_pinhole_camera_parameters("/home/user/code/vlmaps/vlmaps/ScreenCamera.json")
    # ctr.convert_from_pinhole_camera_parameters(parameters)

    # Create a point cloud and visualize it
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    # o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

    # Get the view control
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # ctr = vis.get_view_control()

    # parameters = o3d.io.read_pinhole_camera_parameters("/home/user/code/vlmaps/vlmaps/ScreenCamera.json")
    # ctr.convert_from_pinhole_camera_parameters(parameters)

    # # Rotate the camera
    # ctr.rotate(90.0, 0.0)  # Rotate by 10 degrees around the x-axis
    # ctr.rotate(0.0, 10.0)  # Rotate by 10 degrees around the y-axis

    # vis.add_geometry(pcd)
    # vis.get_render_option().point_size = 3  # Adjust point size for better visualization
    # vis.run()  # Start the visualization thread



def visualize_rgb_map_3d_instances(pc: np.ndarray, rgb: np.ndarray, contours: list, centers: list, bboxs: list, mask):
    grid_rgb = rgb / 255.0

    clouds = []

    # pcd = o3d.geometry.PointCloud()
    # print("pc:", pc.shape, pc.dtype)
    # pcd.points = o3d.utility.Vector3dVector(pc)
    # pcd.colors = o3d.utility.Vector3dVector(grid_rgb)
    # clouds.append(pcd)

    z = 20

    # for contour in contours:
    #     contour = np.array(contour)
    #     newC = np.full((contour.shape[0], 3),z)
    #     newC[:,0:2] = contour
    #     contour = newC
    #     cloud = o3d.geometry.PointCloud()
    #     cloud.points = o3d.utility.Vector3dVector(contour)
    #     cloud.paint_uniform_color([1, 0, 0])
    #     clouds.append(cloud)

    # for center in centers:
    #     center = np.array(center)
    #     newC = np.full((center.shape[0], 3),z)
    #     newC[:, 0:2] = center
    #     center = newC
    #     cloud = o3d.geometry.PointCloud()
    #     cloud.points = o3d.utility.Vector3dVector(center)
    #     cloud.paint_uniform_color([0, 1, 0])
    #     clouds.append(cloud)

    # for bbox in bboxs:
    #     point1 = [bbox[0], bbox[2]]
    #     point2 = [bbox[1], bbox[2]]
    #     point3 = [bbox[0], bbox[3]]
    #     point4 = [bbox[1], bbox[3]]

    #     bbox = np.array([point1, point2, point3, point4])
    #     newC = np.full((bbox.shape[0], 3), z)
    #     newC[:, 0:2] = bbox
    #     bbox = newC
    #     cloud = o3d.geometry.PointCloud()
    #     cloud.points = o3d.utility.Vector3dVector(bbox)
    #     cloud.paint_uniform_color([1, 1, 0])
    #     clouds.append(cloud)

    minx = np.min(pc[:, 0])
    miny = np.min(pc[:, 1])

    masked = []
    masked_col = []
    for i in tqdm(range(pc.shape[0])):
        xyz = pc[i]
        xi = xyz[0]-minx
        yi = xyz[1]-miny
        if(mask[xi,yi]):
            masked.append(xyz)
            masked_col.append(grid_rgb[i])

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(masked)
    cloud.colors = o3d.utility.Vector3dVector(masked_col)
    #cloud.paint_uniform_color([1, 0, 0])
    clouds.append(cloud)

    o3d.visualization.draw_geometries(clouds)


def get_actual_distance_heatmap(mask, scores):
    scores = scores[mask].reshape(-1)
    return heat

def get_heatmap_from_mask_3d(
    pc: np.ndarray, mask: np.ndarray, cell_size: float = 0.05, decay_rate: float = 0.01
) -> np.ndarray:
    target_pc = pc[mask, :]
    other_ids = np.where(mask == 0)[0]
    other_pc = pc[other_ids, :]

    target_sim = np.ones((target_pc.shape[0], 1))
    other_sim = np.zeros((other_pc.shape[0], 1))
    pbar = tqdm(other_pc, desc="Computing heat", total=other_pc.shape[0])
    for other_p_i, p in enumerate(pbar):
        dist = np.linalg.norm(target_pc - p, axis=1) / cell_size
        min_dist_i = np.argmin(dist)
        min_dist = dist[min_dist_i]
        other_sim[other_p_i] = np.clip(1 - min_dist * decay_rate, 0, 1)

    new_pc = pc.copy()
    heatmap = np.ones((new_pc.shape[0], 1), dtype=np.float32)
    for s_i, s in enumerate(other_sim):
        heatmap[other_ids[s_i]] = s
    return heatmap.flatten()


def visualize_masked_map_3d(pc: np.ndarray, mask: np.ndarray, rgb: np.ndarray, transparency: float = 0.5, name: str = "Default"):
    heatmap = mask.astype(np.float16)
    visualize_binary_heatmap_3d(pc, heatmap, rgb, transparency, name)

def log_color_mapping(value):
    # Interpolate between green and yellow
    red = np.log10(value * 12 + 1) / np.log10(10)
    green = np.log10((1-value) * 12 + 1) / np.log10(10)

    # Scale to 0-1 range
    red = min(max(red, 0), 1)
    green = min(max(green, 0), 1)
    if (red == 1 or green == 1):
        blue = 0
    else:
        blue = 0
    blue = min(max(blue, 0), 1)

    return red, green, blue


def exp_color_mapping(value):
    # Apply exponential transformation to the value

    base = 1e6  # You can adjust the base as needed

    tipping = 0.5
    bias = 0.025
    red = (base ** value - 1) / (base - 1)
    blue = 1-(base ** (value+tipping) - 1) / (base - 1)
    if (value > tipping):
        green = 1-(base ** (value-bias) - 1) / (base - 1)
    else:
        green = 1-(base ** (1-value-bias) - 1) / (base - 1)

    # Scale to 0-1 range
    red = min(max(red, 0), 1)
    green = min(max(green, 0), 1)
    blue = min(max(blue, 0), 1)

    return red, green, blue


def clamp(val, lb, ub):
    return min(max(val, lb), ub)


def lin_color_mapping(value):
    # blue = clamp((2*abs(value-0.5))**2,0,1)
    # blue = 2*abs(value-0.5)
    blue = clamp(1-value, 0, 1)**2

    bias = 0.0
    shift = 0.25

    red = clamp(2.0 * value + bias - shift, 0, 1)
    green = clamp((2.0+bias+shift) - 2.0 * value, 0, 1)
    # red, green, blue = red * 255, green * 255, blue * 255
    return red, green, blue

# '#1f77b4', 0.122, 0.467, 0.706 - blue
# '#ff7f0e', 1.000, 0.498, 0.055 - orange
# '#2ca02c', 0.173, 0.627, 0.173 - green
# '#d62728', 0.839, 0.153, 0.157 - red
# '#9467bd', 0.580, 0.404, 0.741 - purple
# '#8c564b', 0.549, 0.337, 0.294 - brown
# '#e377c2', 0.890, 0.467, 0.761 - pink
# '#7f7f7f', 0.498, 0.498, 0.498 - grey
# '#bcbd22', 0.737, 0.741, 0.133 - piss
# '#17becf', 0.090, 0.745, 0.812 - cyan

def visualize_binary_heatmap_3d(pc: np.ndarray, heatmap: np.ndarray, rgb: np.ndarray, transparency: float = 0.5, name: str = "Default"):
    heat = np.zeros((heatmap.shape[0], 3), dtype=np.float32)
    heat_rgb = np.zeros_like(rgb)
    for i in range(heatmap.shape[0]):
        value = heatmap[i]
        orig_col = rgb[i, :]
        # red = orig_col[0]
        # green = orig_col[1]
        # blue = orig_col[2]
        if value == 0:
            tp = 0.75
            heat_rgb[i, :] = orig_col * tp + np.array([219, 0, 7]) * (1-tp)
        else:
            tp = 0
            heat_rgb[i, :] = orig_col * tp + np.array([19, 136, 8]) * (1-tp)

        # if(value == 0):
        #     # red = 0.122
        #     # green = 0.467
        #     # blue = 0.706

        # else:
        #     # red = 0.839
        #     # green = 0.153
        #     # blue = 0.157
        #     red = 1.000
        #     green = 0.498
        #     blue = 0.055

        # red, green, blue = red * 255, green * 255, blue * 255
        # heat[i, 0] = red
        # heat[i, 1] = green
        # heat[i, 2] = blue

    # transparency
    # heat_rgb = heat * transparency + rgb * (1 - transparency)

    visualize_rgb_map_3d(pc, heat_rgb, name)

def visualize_heatmap_3d(pc: np.ndarray, heatmap: np.ndarray, rgb: np.ndarray, transparency: float = 0.5, name: str = "Default"):
    # original
    # sim_new = (heatmap * 255).astype(np.uint8)
    # heat = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
    # heat = heat.reshape(-1, 3)[:, ::-1].astype(np.float32)

    # same as openscene
    print(np.min(heatmap))
    print(np.max(heatmap))
    print(np.mean(heatmap))

    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / np.max(heatmap)
    # if(name=="OpenSeg"):
    #     heatmap = 1 - heatmap

    print(np.min(heatmap))
    print(np.max(heatmap))
    print(np.mean(heatmap))

    heat = np.zeros((heatmap.shape[0], 3),dtype=np.float32)
    for i in range(heatmap.shape[0]):
        value = heatmap[i]
        # red, green, blue = exp_color_mapping(value)
        # red, green, blue = log_color_mapping(value)
        red, green, blue = lin_color_mapping(value)
        red, green, blue = red * 255, green * 255, blue * 255
        heat[i,0] = red
        heat[i,1] = green
        heat[i,2] = blue

    # transparency
    heat_rgb = heat * transparency + rgb * (1 - transparency)

    visualize_rgb_map_3d(pc, heat_rgb, name)


def pool_3d_label_to_2d(mask_3d: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    mask_2d = np.zeros((gs, gs), dtype=bool)
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        mask_2d[row, col] = mask_3d[i] or mask_2d[row, col]

    return mask_2d


def pool_3d_rgb_to_2d(rgb: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    rgb_2d = np.zeros((gs, gs, 3), dtype=np.uint8)
    height = -100 * np.ones((gs, gs), dtype=np.int32)
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        if h > height[row, col]:
            rgb_2d[row, col] = rgb[i]

    return rgb_2d


def get_heatmap_from_mask_2d(mask: np.ndarray, cell_size: float = 0.05, decay_rate: float = 0.01) -> np.ndarray:
    dists = distance_transform_edt(mask == 0) / cell_size
    tmp = np.ones_like(dists) - (dists * decay_rate)
    heatmap = np.where(tmp < 0, np.zeros_like(tmp), tmp)

    return heatmap


def visualize_rgb_map_2d(rgb: np.ndarray):
    """visualize rgb image

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
    """
    rgb = rgb.astype(np.uint8)
    bgr = rgb[:, :, ::-1]
    windowName = "rgb map"
    cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(windowName, 1900, 1900)
    cv2.imshow(windowName, bgr)
    cv2.waitKey(0)


def visualize_heatmap_2d(rgb: np.ndarray, heatmap: np.ndarray, transparency: float = 0.5):
    """visualize heatmap

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
        heatmap (np.ndarray): (gs, gs) element range [0, 1] np.float32
    """
    sim_new = (heatmap * 255).astype(np.uint8)
    heat = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
    heat = heat[:, :, ::-1].astype(np.float32)  # convert to RGB
    heat_rgb = heat * transparency + rgb * (1 - transparency)
    visualize_rgb_map_2d(heat_rgb)


def visualize_masked_map_2d(rgb: np.ndarray, mask: np.ndarray):
    """visualize masked map

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
        mask (np.ndarray): (gs, gs) element range [0, 1] np.uint8
    """
    visualize_heatmap_2d(rgb, mask.astype(np.float32))


def create_3d_map_from_mask(mask: np.ndarray, xyz: np.ndarray):
    xmin = np.min(xyz[:, 0])
    ymin = np.min(xyz[:, 1])
    zmin = np.min(xyz[:, 2])
    xmax = np.max(xyz[:, 0]) - xmin
    ymax = np.max(xyz[:, 1]) - ymin
    zmax = np.max(xyz[:, 2]) - zmin

    grid = np.zeros((xmax + 1, ymax + 1, zmax + 1))

    for i, pos in enumerate(xyz):
        x, y, z = pos[0] - xmin, pos[1] - ymin, pos[2] - zmin
        grid[x, y, z] = mask[i]
    return grid

def create_2d_map_from_mask(mask: np.ndarray, xyz: np.ndarray):
    mask_3d = create_3d_map_from_mask(mask, xyz)
    return mask_3d.max(axis=2).astype(dtype=np.uint8)

