from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
import numpy as np
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
)
from vlmaps.map.vlmap import MapType
from tqdm import tqdm
import warnings
from methods.common.factory import get_default_embedders
from methods.method import Method, TaskType, SynonymType
from methods.common.creators.visualEncoderFactory import EncoderType
from methods.parameter_tuning.parameter_tuner import ParameterTuner, OptimizationObjective
import utils.classification as classification
import utils.common as common
from utils.postprocessing import postprocess_3d
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
from wonderwords import RandomWord

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="parameter_tuning.yaml",
)
def main(config: DictConfig) -> None:
    encoder_name = config.map_config.visual_encoder.name
    classifier = config.classifier
    metric = config.metric

    warnings.simplefilter("ignore")
    classes, labels, names, num_classes_orig = common.parseClassFile(config.classes, config.delimiter)

    # Load maps
    # gt = VLMap(config.map_config, data_dir=data_dirs[id])
    # gt_success = gt.load_map_override(data_dirs[id], config.direct_path_path)
    # assert gt_success, f"Map loading failed from {config.direct_path_path}"

    # vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    # vlmap_success = vlmap.load_map_override(data_dirs[id], config.direct_path_path)
    # assert vlmap_success, f"Map loading failed from {config.direct_path_path}"

    # create optimizer
    if not config.experiment:
        experiment = f"{new_name()}_{encoder_name.lower()}_{classifier}_{metric}"
    else:
        experiment = config.experiment
    if encoder_name.lower() == "lseg":
        encoder = EncoderType.LSEG
    else:
        encoder = EncoderType.OPENSEG
    temp_folder = "./../optimize-temp/"
    task = TaskType.SEGMENTATION
    num_trials = config.num_trials
    multiprocessing = 1
    no_save = False
    start_state = "default"
    objective = OptimizationObjective.PARAMS if config.type == "params" else OptimizationObjective.SYNONYMS
    tuner = ParameterTuner(
        encoder,
        classifier,
        objective,
        temp_folder,
        task, experiment, num_trials, multiprocessing, no_save, start_state, config.use_prompt_engineering, SynonymType(config.synonym_type), antonym_ratio=config.antonym_ratio
    )

    tuner.optimize(
        cost_function,
        config=config,
        labels=labels,
        names=names,
        metric=metric,
        gs=config.params.gs
    )
    tuner.write_results()

def new_name() -> str:
    r = RandomWord()
    adj = r.word(include_parts_of_speech=["adjectives"])
    noun = r.word(include_parts_of_speech=["noun"])
    return f"{adj}_{noun}"

def cost_function(method, config, labels, names, metric, gs) -> float:
    if config.scene_id >= 0:
        ids = [config.scene_id]
    else:
        ids = list(range(10))

    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([str(x) for x in data_dir.iterdir() if x.is_dir()])

    classifications = []
    for id in ids:
        if config.sample_map:
            maptype = MapType.SAMPLE
        else:
            maptype = MapType.REGULAR
        # GT
        gt = VLMap(config.map_config, data_dir=data_dirs[id])
        gt_success = gt.load_map(data_dirs[id], maptype)
        assert gt_success, "Map loading failed"

        # VLMAPS have to be custom compared because they don't segment the image
        vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
        vlmap_success = vlmap.load_map(data_dirs[id], maptype)
        assert vlmap_success, "Map loading failed"

        vlmap._init_clip(config.map_config.visual_encoder.vlm_version)
        vlmap.init_categories(mp3dcat[1:-1], config.use_prompt_engineering)
        obstacle_map = vlmap.generate_obstacle_map()
        _ = vlmap.generate_cropped_obstacle_map(obstacle_map)

        for i in tqdm(range(len(labels)), desc="vlmap labels"):
            name = names[i]
            label = int(labels[i])

            # create GT
            true = (gt.grid_semantic == label).squeeze()

            # ignore empty classes
            # if np.sum(true) < 100:
            #     continue

            mask, _, _, _, _, _, _ = density_query(vlmap, name, "other", method, False, gs, is_3d=True)
            pred = mask.astype(np.bool_)

            # classify
            cf = classification.classify_all(true, pred, label)
            classifications.append(cf)
    if metric == "f1":
        result = classification.micro_f1(classifications)
    elif metric == "iou":
        result = classification.micro_iou(classifications)
    else:
        raise ValueError("Unknown metric")
    return result

def create_1d_from_2d(mask: np.ndarray, vlmap: VLMap) -> np.ndarray:
    minx = np.min(vlmap.grid_pos[:, 0])
    miny = np.min(vlmap.grid_pos[:, 1])

    # masked area
    mask_1d = []
    for i in range(vlmap.grid_pos.shape[0]):
        xyz = vlmap.grid_pos[i]
        xi = xyz[0]-minx
        yi = xyz[1]-miny

        # In some cases, the grid_pos exceeds the size of the mask
        # and causes index out of bounds exception
        if (xi < mask.shape[0] and yi < mask.shape[1] and mask[xi, yi]):
            mask_1d.append(True)
        else:
            mask_1d.append(False)
    pred = np.array(mask_1d).astype(np.bool_)
    return pred

def density_query(vlmap: VLMap, text: str, complement: str, method: Method, postprocessing: bool, gs: int, is_3d: bool):
    synonyms, complement_synonyms = method.get_synonyms(text, [complement])
    pc_mask = method.predict(vlmap.grid_feat, text, [complement], synonyms=synonyms, complement_synonyms=complement_synonyms)
    mask_2d = pool_3d_label_to_2d(pc_mask, vlmap.grid_pos, gs)
    mask_2d = mask_2d[vlmap.rmin : vlmap.rmax + 1, vlmap.cmin : vlmap.cmax + 1]

    if is_3d:
        if postprocessing:
            mask = postprocess_3d(pc_mask, vlmap.grid_pos, gs)
        else:
            mask = pc_mask
        postprocessing_1 = np.zeros_like(mask_2d)
        postprocessing_2 = np.zeros_like(mask_2d)
        postprocessing_3 = np.zeros_like(mask_2d)
        postprocessing_4 = np.zeros_like(mask_2d)
    else:
        if postprocessing:
            postprocessing_1 = binary_closing(mask_2d, iterations=3)
            postprocessing_2 = gaussian_filter(postprocessing_1.astype(float), sigma=0.8, truncate=3)
            postprocessing_3 = postprocessing_2 > 0.5
            postprocessing_4 = binary_dilation(postprocessing_3)
            mask = postprocessing_4
        else:
            postprocessing_1 = np.zeros_like(mask_2d)
            postprocessing_2 = np.zeros_like(mask_2d)
            postprocessing_3 = np.zeros_like(mask_2d)
            postprocessing_4 = np.zeros_like(mask_2d)
            mask = mask_2d

    return mask, pc_mask, mask_2d, postprocessing_1, postprocessing_2, postprocessing_3, postprocessing_4

def dbg_to_2d(array_1d, grid_):
    xmin = np.min(grid_[:, 0])
    ymin = np.min(grid_[:, 1])
    dx = np.max(grid_[:, 0]) - xmin
    dy = np.max(grid_[:, 1]) - ymin
    grid_size_ = max(dx, dy) + 1
    output = np.zeros((grid_size_, grid_size_), dtype=bool)
    for i, pos in enumerate(grid_):
        row, col, h = pos
        row, col = row - xmin, col - ymin
        output[row, col] = array_1d[i] or output[row, col]
    return output

if __name__ == "__main__":
    main()
