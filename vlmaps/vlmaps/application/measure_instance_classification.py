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
from sklearn import metrics
import warnings

import utils.common as common
import utils.classification as classification
import utils.result_writing as results
import os
from tqdm import tqdm


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="measure_instance_classification.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    id = config.scene_id
    batch = config.batch
    out_path = config.output_path
    if config.use_prompt_engineering:
        out_path = out_path.replace("analysis", "analysis-prompt-engineering")

    warnings.simplefilter("ignore")

    classes, labels, names, num_classes_orig = common.parseClassFile(config.classes, config.delimiter)
    names = names[1:-1]
    labels = labels[1:-1]

    # prepare output
    fileName = f"{out_path}/queryability_instances_{config.map_config.map_prefix}_{id}_{config.map_config.cell_size}.csv"
    file_exists = os.path.exists(fileName)
    force = config.force
    if file_exists:
        if not force:
            print("Analysis exists, exiting")
            exit(0)
        else:
            print("force = True, overwriting exiting analysis")
            os.remove(fileName)

    cl_out_file = results.getFile(out_path, f"queryability_instances_{config.map_config.map_prefix}_{id}_{config.map_config.cell_size}", "csv")
    results.writeClassificationHeader(cl_out_file)
    # debug output
    cl_dbg_file = results.getFile(out_path, f"queryability_instances_debug_{config.map_config.map_prefix}_{id}_{config.map_config.cell_size}", "csv")
    header = "map;id;tp;tn;fp;fn;i;u"
    results.writeString(cl_dbg_file, header)

    # GT
    gt = VLMap(config.map_config, data_dir=data_dirs[id])
    gt.load_map(data_dirs[id], MapType.INSTANCES)
    u_instances = np.unique(gt.grid_instance)

    ############################################################################
    # VLMAPS have to be custom compared because the instance segmentation suck
    vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    vlmap.load_map(data_dirs[id], MapType.PREDICTED)
    vlmap._init_clip(config.map_config.visual_encoder.vlm_version)
    vlmap.init_categories(mp3dcat[1:-1], config.use_prompt_engineering)
    obstacle_map = vlmap.generate_obstacle_map()
    _ = vlmap.generate_cropped_obstacle_map(obstacle_map)

    classifications = []
    for i in tqdm(range(len(labels))):
        name = names[i]
        label = int(labels[i])

        contours, centers, bbox_list, mask, pc_mask, mask_2d, postprocessed_1, postprocessed_2, postprocessed_3, postprocessed_4 = vlmap.get_pos(name, config.use_prompt_engineering)
        # no instances
        # if(contours is None or len(contours) == 0):
        #     continue

        minx = np.min(vlmap.grid_pos[:, 0])
        miny = np.min(vlmap.grid_pos[:, 1])

        # masked area
        mask_3d = []
        for i in range(vlmap.grid_pos.shape[0]):
            xyz = vlmap.grid_pos[i]
            xi = xyz[0]-minx
            yi = xyz[1]-miny

            # In some cases, the grid_pos exceeds the size of the mask
            # and causes index out of bounds exception
            if (xi < mask.shape[0] and yi < mask.shape[1] and mask[xi, yi]):
                mask_3d.append(True)
            else:
                mask_3d.append(False)
        pred = np.array(mask_3d)

        y_true = []
        y_pred = []
        for instance_id in u_instances:
            idx = (gt.grid_instance == instance_id).squeeze()
            preds = pred[idx]
            gt_semantics = gt.grid_semantic[idx]
            gu, gc = np.unique(gt_semantics, return_counts=True)
            true_label = gu[np.argmax(gc)]

            uniques, counts = np.unique(preds, return_counts=True)
            majority = uniques[np.argmax(counts)]

            y_true.append(true_label == label)
            y_pred.append(majority)

        y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred)
        cf = classification.classify_all(y_true, y_pred, label)
        classifications.append(cf)

    out = classification.aggregate(id, "vlmaps", classifications, False)
    results.writeClassificationLine(cl_out_file, [out])

    if classifications:
        for cls in classifications:
            line = str(id) + ";" + cls.tostring()
            results.writeString(cl_dbg_file, line)

    ############################################################################
    # COMPARISONS
    predictions = []

    # our instances
    comp = VLMap(config.map_config, data_dir=data_dirs[id])
    comp.load_map(data_dirs[id], MapType.OUR_INSTANCES)
    predictions.append((comp.grid_semantic, "our_instances"))

    # predicted semantics
    comp = VLMap(config.map_config, data_dir=data_dirs[id])
    comp.load_map(data_dirs[id], MapType.PREDICTED)
    predictions.append((comp.grid_semantic, "predicted"))

    # predicted postprocessed semantics
    # comp = VLMap(config.map_config, data_dir=data_dirs[id])
    # comp.load_map(data_dirs[id], MapType.PREDICTED_POSTPROCESSED)
    # predictions.append((comp.grid_semantic, "predicted_postprocessed"))

    # for (pred, name) in predictions:
    #     classifications = []
    #     y_true = []
    #     y_pred = []
    #     for instance_id in u_instances:
    #         idx = (gt.grid_instance == instance_id).squeeze()
    #         true_label = gt.grid_semantic[idx][0]
    #         preds = pred[idx]

    #         uniques, counts = np.unique(preds, return_counts=True)
    #         majority = uniques[np.argmax(counts)]

    #         y_true.append(true_label)
    #         y_pred.append(majority)

    #     y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred)
    #     cf = classification_metrics.classify_all(y_true, y_pred, 0)
    #     classifications.append(cf)
    #     out = classification_metrics.aggregate(id, name, classifications, False)
    out, classifications, (y_true, y_pred) = classification.instance_classification(
        predictions, gt.grid_semantic, gt.grid_instance, labels, id)
    results.writeClassificationLine(cl_out_file, out)

    if classifications:
        for cls in classifications:
            line = str(map) + ";" + cls.tostring()
            results.writeString(cl_dbg_file, line)


if __name__ == "__main__":
    main()
