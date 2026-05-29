from pathlib import Path
from typing import Optional, List
import hydra
from omegaconf import DictConfig
from itertools import product
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
    create_2d_map_from_mask
)
from vlmaps.map.vlmap import MapType
from tqdm import tqdm
from sklearn import metrics
import os
import warnings
from methods.common.factory import get_default_embedders
from methods.common.embedders import TextEmbedder, ImageEmbedder
from methods.method import Method, TaskType, SynonymType
from methods.common.creators.visualEncoderFactory import EncoderType

import utils.result_writing as results
import utils.classification as classification
import utils.common as common
from utils.postprocessing import postprocess_3d
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
import random


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="measure_classification.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    encoder_name = config.map_config.visual_encoder.name
    if encoder_name == "LSeg":
        encoder_type = EncoderType.LSEG
    else:
        encoder_type = EncoderType.OPENSEG

    classifier = config.classifier
    svm_config = get_svm_config(config.classifier_config, classifier, encoder_name.lower())
    if config.classifier_config is not None:
        classifier_config_filename = Path(config.classifier_config).name.replace(".yaml", "")
    else:
        if svm_config:
            classifier_config_filename = Path(svm_config).name.replace(".yaml", "")
        else:
            classifier_config_filename = "default"

    id = config.scene_id
    out_path = config.output_path
    is_density = config.density
    use_3d = not config.index_2d
    use_prompt_engineering = config.prompt_engineering > 0
    debug = config.debug
    variable_config = config.variable_config
    if config.experiment is not None:
        out_path = f"{out_path}/{config.experiment}"
    else:
        if config.named_experiment:
            out_path = results.createNamedExperiment(out_path, is_density, encoder_name.lower(), verbose=True)

    density_initialized = False
    image_embedder: Optional[ImageEmbedder] = None
    text_embedder: Optional[TextEmbedder] = None
    method: Optional[Method] = None

    warnings.simplefilter("ignore")
    force = config.force

    extra_dbg_str = f"-{config.metadata}"
    if use_prompt_engineering:
        extra_dbg_str += "-promptengineering"
    if config.use_postprocessing:
        extra_dbg_str += "-postprocessing"
    if config.median:
        extra_dbg_str += "-median"
    if not use_3d:
        extra_dbg_str += "-2d"
    if is_density:
        extra_dbg_str += "-density"
    if variable_config:
        extra_dbg_str += f"-{classifier_config_filename}"

    cl_out_file = results.getFile(
        out_path,
        f"queryability_{config.map_config.map_prefix}_{id}_{config.map_config.cell_size}",
        "csv",
        exist_ok=True
    )
    if not Path(cl_out_file).exists():
        results.writeClassificationHeader(cl_out_file)

    cl_dbg_file = results.getFile(
        out_path,
        f"debug_{config.map_config.map_prefix}_{id}_{config.map_config.cell_size}{extra_dbg_str}",
        "csv",
        force,
    )
    header = "map;id;tp;tn;fp;fn;i;u"
    results.writeString(cl_dbg_file, header)

    classes, labels, names, num_classes_orig = common.parseClassFile(config.classes, config.delimiter, config.remove_aggregate_classes)

    dbg_save_dir = f"/home/user/Desktop/ICRA-image-debug/debug-vlmaps-{config.map_config.map_prefix}-{config.map_config.cell_size}{extra_dbg_str}/{id}"
    Path(dbg_save_dir).mkdir(exist_ok=True, parents=True)

    maptype = MapType.REGULAR
    if config.median:
        maptype = MapType.MEDIAN
    if config.sample:
        maptype = MapType.SAMPLE
    elif config.transformer is not None:
        if config.transformer == "new":
            maptype = MapType.TRANSFORMER_NEW
        elif config.transformer == "lora":
            maptype = MapType.TRANSFORMER_LORA
        else:
            raise ValueError("Unknown transformer type")
    # GT
    gt = VLMap(config.map_config, data_dir=data_dirs[id])
    gt_success = gt.load_map(data_dirs[id], maptype, encoder_name.lower())
    assert gt_success, f"Map loading failed"

    # VLMAPS have to be custom compared because they don't segment the image
    vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    vlmap_success = vlmap.load_map(data_dirs[id], maptype, encoder_name.lower())
    assert vlmap_success, "Map loading failed"

    vlmap._init_clip(config.map_config.visual_encoder.vlm_version)

    # FIXME: debug
    # vlmap.init_categories(mp3dcat[1:-1], use_prompt_engineering)
    vlmap.categories = mp3dcat[1:-1]

    obstacle_map = vlmap.generate_obstacle_map()
    _ = vlmap.generate_cropped_obstacle_map(obstacle_map)

    classifications = []
    all_predictions = np.zeros((num_classes_orig, vlmap.grid_feat.shape[0]))
    all_gts = np.zeros_like(all_predictions)

    complement_list = None
    if config.complement is not None:
        complement_list = config.complement.split(",")
        print(f"Using complements: {complement_list}")

    for i in tqdm(range(len(labels)), desc="vlmap labels"):
        # FIXME: debug
        # if i > 1:
        #     break
        name = names[i]
        label = int(labels[i])

        if debug:
            method_mask = None
            vlmaps_mask = None

        # run_density = is_density or debug
        # run_vlmaps = debug or not is_density
        run_density = is_density
        run_vlmaps = not is_density

        pred: np.ndarray = np.array(())
        mask: np.ndarray = np.array(())
        mask_2d: np.ndarray = np.array(())
        postprocessed_1: np.ndarray = np.array(())
        postprocessed_2: np.ndarray = np.array(())
        postprocessed_3: np.ndarray = np.array(())
        postprocessed_4: np.ndarray = np.array(())

        if run_density:
            if not density_initialized:
                method = Method(
                    task=TaskType.SEGMENTATION,
                    encoder=encoder_type,
                    classifier=classifier,
                    classifier_config_path=svm_config,
                    prompt_engineering=config.prompt_engineering,
                    prompts=config.prompts,
                    synonym_type=SynonymType(config.synonym_type),
                    antonym_type=SynonymType(config.antonym_type)
                )
                image_embedder, text_embedder = get_default_embedders(encoder_type)
                assert image_embedder
                assert text_embedder
                method.set_embedders(image_embedder, text_embedder)
                density_initialized = True
                # print(method.get_method_description(True))
            assert method
            try:
                sample_synonyms = config.sample_synonyms
            except:
                sample_synonyms = -1
            if classifier == "nn":
                mask, pc_mask, mask_2d, postprocessed_1, postprocessed_2, postprocessed_3, postprocessed_4 = nn_query(
                    vlmap,
                    name,
                    method,
                    config.use_postprocessing,
                    config.params.gs,
                    use_3d,
                    names,
                    labels
                )
            else:
                mask, pc_mask, mask_2d, postprocessed_1, postprocessed_2, postprocessed_3, postprocessed_4 = density_query(
                    vlmap,
                    name,
                    "other",
                    method,
                    config.use_postprocessing,
                    config.params.gs,
                    is_3d=use_3d,
                    sample_synonyms=sample_synonyms,
                    no_synonyms=config.no_synonyms
                )
            if not use_3d:
                pred = create_1d_from_2d(mask, vlmap).astype(np.bool_)
            else:
                pred = mask.astype(np.bool_)

            if debug:
                method_mask = mask
        if run_vlmaps:
            if use_3d:
                mask, _ = vlmap.index_map(name, use_prompt_engineering, with_init_cat=False, add_other=True, complement=complement_list)
                if config.use_postprocessing:
                    mask = postprocess_3d(mask, vlmap.grid_pos, config.params.gs).astype(np.bool_)

                pc_mask = np.zeros_like(mask)
                mask_2d = np.zeros_like(mask)
                postprocessed_1 = np.zeros_like(mask)
                postprocessed_2 = np.zeros_like(mask)
                postprocessed_3 = np.zeros_like(mask)
                postprocessed_4 = np.zeros_like(mask)
                pred = mask
            else:
                _, _, _, mask, pc_mask, mask_2d, postprocessed_1, postprocessed_2, postprocessed_3, postprocessed_4 = (
                    vlmap.get_pos(name, use_prompt_engineering, config.use_postprocessing, with_init_cat=False)
                )
                pred = create_1d_from_2d(mask, vlmap)

            if debug:
                vlmaps_mask = mask
        # if debug:
        #     fig, ax = plt.subplots(1, 3)
        #     vm = dbg_to_2d(vlmaps_mask, vlmap.grid_pos).astype(dtype=np.uint8)  # type: ignore
        #     mm = dbg_to_2d(method_mask, vlmap.grid_pos).astype(dtype=np.uint8)  # type: ignore
        #     ax[0].imshow(vm)
        #     ax[1].imshow(mm)
        #     ax[2].imshow(vm-mm)
        #     plt.show()

        # create GT
        if np.any(gt.grid_semantic == label):
            true = (gt.grid_semantic == label).squeeze()
        else:
            true = np.full_like(pred, False)

        # classify
        assert true.shape == pred.shape
        cf = classification.classify_all(true, pred, label)
        classifications.append(cf)

        # count hits/misses
        all_predictions[label, :] = pred
        all_gts[label, :] = true

        line = str(id) + ";" + cf.to_csv()
        results.writeString(cl_dbg_file, line)

        # FIXME: debug
        if debug:
            dbg_mask = create_2d_map_from_mask(mask, vlmap.grid_pos)
            fig, ax = plt.subplots(1, 2, figsize=(15, 15))
            fig.suptitle(name)
            ax[0].imshow(dbg_mask.astype(dtype=np.uint8))
            ax[0].set_title("mask_2d")
            ax[1].imshow(dbg_to_2d(true, vlmap.grid_pos).astype(dtype=np.uint8))
            ax[1].set_title("gt_mask")
            fig.savefig(f"{dbg_save_dir}/{name}.png")
            plt.close(fig)

    all_predictions_file = f"{out_path}/all-predictions-{encoder_name.lower()}-{id}{extra_dbg_str}"
    all_gts_file = f"{out_path}/all-gts-{encoder_name.lower()}-{id}{extra_dbg_str}"
    np.save(all_predictions_file, all_predictions)
    np.save(all_gts_file, all_gts)

    grid_file = f"{out_path}/grid-{id}"
    if not Path(grid_file).exists():
        np.save(grid_file, vlmap.grid_pos)

    if is_density:
        title = "method"
    else:
        title = "vlmaps"
    out = classification.aggregate(id, title, classifications, False)
    optional_metadata = config.metadata if config.metadata is not None else ""
    metadata = [use_prompt_engineering, config.use_postprocessing, config.median, not use_3d, is_density, classifier_config_filename, config.prompts, config.synonym_type, config.antonym_type, config.classifier, encoder_name.lower(), optional_metadata]
    results.writeClassificationLine(cl_out_file, [out], [metadata])

    if config.verbose_experiment:
        results.print_classification_line([out])

    common_metadata = results.getFile(out_path, f"run_config_{encoder_name}_{id}{extra_dbg_str}", "out")
    with open(common_metadata, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    if is_density:
        assert method
        cl_metadata = results.getFile(
            out_path, f"density_metadata_{encoder_name}_{id}{extra_dbg_str}", "out"
        )
        with open(cl_metadata, "w") as f:
            f.write(f"use_prompt_engineering: {use_prompt_engineering}\n")
            f.write(f"use_postprocessing: {config.use_postprocessing}\n")
            f.write(f"use_3d: {use_3d}\n")
            f.write(method.get_method_description() + "\n")


def create_1d_from_2d(mask: np.ndarray, vlmap: VLMap) -> np.ndarray:
    minx = np.min(vlmap.grid_pos[:, 0])
    miny = np.min(vlmap.grid_pos[:, 1])

    # diffx = np.nonzero(vlmap.grid_pos[:, 0] - minx > mask.shape[0])[0].shape[0]
    # diffy = np.nonzero(vlmap.grid_pos[:, 1] - miny > mask.shape[1])[0].shape[0]
    # print(f"diffx: {diffx}")
    # print(f"diffy: {diffy}")

    # masked area
    mask_1d = []
    for i in range(vlmap.grid_pos.shape[0]):
        xyz = vlmap.grid_pos[i]
        xi = xyz[0] - minx
        yi = xyz[1] - miny

        # In some cases, the grid_pos exceeds the size of the mask
        # and causes index out of bounds exception
        if xi < mask.shape[0] and yi < mask.shape[1] and mask[xi, yi]:
            mask_1d.append(True)
        else:
            mask_1d.append(False)
    pred = np.array(mask_1d).astype(np.bool_)
    return pred


def density_query(
    vlmap: VLMap,
    text: str,
    complement: str,
    method: Method,
    postprocessing: bool,
    gs: int,
    is_3d: bool,
    sample_synonyms: int = -1,
    no_synonyms: bool = False
):
    if no_synonyms:
        synonyms: List[str] = []
        complement_synonyms: List[str] = []
    else:
        synonyms, complement_synonyms = method.get_synonyms(text, complement)
        if sample_synonyms > 0 and sample_synonyms < len(synonyms):
            synonyms.remove(text)
            synonyms = random.sample(synonyms, sample_synonyms)
            synonyms.append(text)
        if sample_synonyms > 0 and sample_synonyms < len(complement_synonyms):
            if "background" in complement_synonyms:
                complement_synonyms.remove("background")
            if "nothing" in complement_synonyms:
                complement_synonyms.remove("nothing")
            complement_synonyms = random.sample(complement_synonyms, sample_synonyms)
            complement_synonyms.append("other")

    # print("synonyms")
    # print(synonyms)
    # print("complement_synonyms")
    # print(complement_synonyms)

    pc_mask = method.predict(
        vlmap.grid_feat, text, [complement], synonyms=synonyms, complement_synonyms=complement_synonyms
    )
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


def nn_query(
    vlmap: VLMap,
    text: str,
    method: Method,
    postprocessing: bool,
    gs: int,
    is_3d: bool,
    names: np.ndarray,
    labels: np.ndarray
):
    if text in names:
        res = np.where(names == text)
        i = res[0].item()
        label = int(labels[i])
        clf = method.get_get_predictor()
        preds = clf.predict(vlmap.grid_feat)
        pc_mask = preds == label
    # only closed vocabulary prediction!
    else:
        pc_mask = np.zeros_like(vlmap.grid_semantic)

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


def get_svm_config(classifier_config: str, classifier: str, encoder: str):
    svm_config = ""

    if classifier_config:
        classifier_config = classifier_config.replace("/home/user", "/home/user")

    paths = [
        classifier_config,
        f"/home/user/<path/to/quash>/methods/config/{classifier_config}",
        f"/home/user/<path/to/quash>/methods/config/{classifier}_{classifier_config}",
        f"/home/user/<path/to/quash>/methods/config/{classifier}_{encoder}_{classifier_config}",
    ]
    suffixes = ["", ".yml", ".yaml"]

    for path, suffix in product(paths, suffixes):
        full_path = f"{path}{suffix}"
        if Path(full_path).exists():
            svm_config = full_path
            break


    return svm_config


if __name__ == "__main__":
    main()  # type: ignore
