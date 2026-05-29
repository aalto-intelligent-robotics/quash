import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter

def test_module():
    print("utils.postprocessing loaded")

def convert_1d_to_3d(mask: np.ndarray, grid_pos: np.ndarray, grid_size: int):
    maxh = np.max(grid_pos[:, 2])
    mask_3d = np.zeros((grid_size, grid_size, maxh+1), dtype=bool)
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        mask_3d[row, col, h] = mask[i]
    return mask_3d

def convert_3d_to_1d(mask_3d: np.ndarray, grid_pos: np.ndarray, orig_len: int):
    mask = np.zeros((orig_len))
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        mask[i] = mask_3d[row, col, h]
    return mask

def postprocess_3d(mask: np.ndarray, grid_pos: np.ndarray, grid_size: int):
    # 1D mask to 3D
    mask_3d = convert_1d_to_3d(mask, grid_pos, grid_size)

    # 3D postprocessing
    foreground = binary_closing(mask_3d, iterations=3)
    foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
    foreground = foreground > 0.5
    foreground = binary_dilation(foreground)

    # binary_mask = foreground == label
    binary_mask = foreground

    # 3D to 1D
    mask_1d = convert_3d_to_1d(binary_mask, grid_pos, mask.shape[0])

    return mask_1d
