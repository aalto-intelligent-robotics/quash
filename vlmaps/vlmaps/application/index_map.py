# needed for debugging
import sys

sys.path.append("/home/user/code/vlmaps/vlmaps")

import random
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
import matplotlib.pyplot as plt
from vlmaps.map.vlmap import VLMap
from vlmaps.utils.matterport3d_categories import mp3dcat
from utils.postprocessing import postprocess_3d
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
    create_2d_map_from_mask,
)


sys.path.insert(0, "/home/user/code/vlmaps/vlmaps/utils/")
from methods.common.factory import get_default_embedders
from methods.method import Method, TaskType, SynonymType, PromptEngineering
from methods.common.creators.visualEncoderFactory import EncoderType
import utils.classification as classification
import utils.common as common


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg_lseg.yaml",
    # config_name="map_indexing_cfg_openseg.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    print(config.map_config.visual_encoder.name)

    encoder = config.map_config.visual_encoder.name

    id = config.scene_id
    print("id", id)
    inp = ""
    vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    if config.direct_path:
        print("direct path loading")
        vlmap.load_map_override(data_dirs[id], config.direct_path_path)
    else:
        vlmap.load_map(data_dirs[id])

    classes, labels, names, num_classes_orig = common.parseClassFile(config.classes, config.delimiter)

    # FIXME
    orig_feats = np.copy(vlmap.grid_feat)
    norms = np.linalg.norm(vlmap.grid_feat, axis=1)
    mask = norms > 0
    norms = norms[:, np.newaxis]
    mean_norm = np.mean(norms)
    normalized_feats = np.copy(vlmap.grid_feat)
    normalized_feats[mask, :] = vlmap.grid_feat[mask, :] / norms[mask, :]
    nnorms = np.linalg.norm(normalized_feats, axis=1)
    mean_normalized_norm = np.mean(nnorms)

    # for i in range(vlmap.grid_feat.shape[0]):
    #     orig = vlmap.grid_feat[i, :]
    #     new = normalized_feats[i, :]
    #     orig_norm = norms[i, :].squeeze()
    #     new_norm = nnorms[i]
    #     sim = np.dot(orig, new)/(orig_norm * new_norm)

    #     test = orig_norm * new
    #     same = np.allclose(orig, test)
    #     print(f"similarity {sim:.4f} --- is same {same}")

    vlmap._init_clip(config.map_config.visual_encoder.vlm_version)

    #is_density, is_comparison = select_mode()
    is_density, is_comparison = True, False

    complement = "other"
    # if is_density or is_comparison:
    #     complement_choice = input("Complement: ")
    #     if complement_choice:
    #         complement = complement_choice

    while inp != "q":
        # ch = input("Normalize? y/n: ")
        # if ch == "y":
        #     vlmap.grid_feat = normalized_feats
        # else:
        #     vlmap.grid_feat = orig_feats

        density_mask = np.array(())
        regular_mask = np.array(())
        visualize_rgb_map_3d(vlmap.grid_pos, vlmap.grid_rgb)
        text = input("Query: ")
        # c_choice = input("Complement: ")
        # if c_choice:
        #     complement = c_choice
        # else:
        #     complement = "other"
        complement = "other"

        if is_density:
            mask = density_query(encoder, vlmap, text, complement, config.params.gs, names, labels)

            visualize_masked_map_3d(vlmap.grid_pos, mask, vlmap.grid_rgb, 1.0, encoder)
        elif is_comparison:
            density_mask = density_query(encoder, vlmap, text, complement, config.params.gs, names, labels)
            regular_mask, scores = vlmaps_query(vlmap, text, config.params.gs)

            density_mask_2d = create_2d_map_from_mask(density_mask, vlmap.grid_pos)
            regular_mask_2d = create_2d_map_from_mask(regular_mask, vlmap.grid_pos)

            fig, ax = plt.subplots(2, 3)
            ax[0, 0].imshow(density_mask_2d)
            ax[0, 0].set_title("Ours")
            ax[0, 1].imshow(regular_mask_2d)
            ax[0, 1].set_title("VLMaps")
            diff = density_mask_2d.astype(dtype=np.int8) - regular_mask_2d.astype(dtype=np.int8)
            ax[0, 2].imshow(diff, cmap="seismic", vmin=-1, vmax=1)
            ax[0, 2].set_title("Diff")

            if text in names:
                idx = list(names).index(text)
                label = int(labels[idx])
                gt_mask = (vlmap.grid_semantic == label).squeeze()
                gt_mask_2d = create_2d_map_from_mask(gt_mask, vlmap.grid_pos)
                ax[1, 2].imshow(gt_mask_2d)

                ax[1, 0].imshow(
                    density_mask_2d.astype(dtype=np.int8) - gt_mask_2d.astype(dtype=np.int8),
                    cmap="seismic",
                    vmin=-1,
                    vmax=1,
                )
                ax[1, 0].set_title("Diff")

                ax[1, 1].imshow(
                    regular_mask_2d.astype(dtype=np.int8) - gt_mask_2d.astype(dtype=np.int8),
                    cmap="seismic",
                    vmin=-1,
                    vmax=1,
                )
                ax[1, 1].set_title("Diff")

            plt.show()
        else:
            mask, scores = vlmaps_query(vlmap, text, config.params.gs)

            if config.index_2d:
                rgb_2d = pool_3d_rgb_to_2d(vlmap.grid_rgb, vlmap.grid_pos, config.params.gs)
                visualize_masked_map_2d(rgb_2d, mask)
                heatmap = get_heatmap_from_mask_2d(mask, cell_size=config.params.cs, decay_rate=config.decay_rate)
                visualize_heatmap_2d(rgb_2d, heatmap)
            else:
                if config.mask_map:
                    visualize_masked_map_3d(vlmap.grid_pos, mask, vlmap.grid_rgb, 1.0, encoder)

            if config.score_map:
                heatmap = scores[:, 0]
                visualize_heatmap_3d(vlmap.grid_pos, heatmap, vlmap.grid_rgb, 1.0, encoder)

        if is_comparison:
            show_results(vlmap, regular_mask, text, names, labels, "regular")
            show_results(vlmap, density_mask, text, names, labels, "density")
        else:
            show_results(vlmap, mask, text, names, labels, "")

        print("c: change mode")
        print("q: quit")
        inp = input(": ")
        if inp == "q":
            exit()
        elif inp == "c":
            is_density, is_comparison = select_mode()
            continue
        else:
            continue


def show_results(vlmap: VLMap, mask: np.ndarray, text: str, names: np.ndarray, labels: np.ndarray, title: str):
    if text in names:
        res = np.where(names == text)
        i = res[0].item()
        label = int(labels[i])
        # create GT
        if np.any(vlmap.grid_semantic == label):
            true = (vlmap.grid_semantic == label).squeeze()
        else:
            true = np.full_like(mask, False)

        # classify
        cf = classification.classify_all(true, mask, label)
        if title:
            print(title)
        print("")
        cf.print()
        print("")


def select_mode():
    is_density = False
    is_comparison = False
    print("Modes:")
    print("d: density")
    print("r: regular")
    print("c: comparison")
    choice = input(": ")
    if choice == "d":
        is_density = True
    elif choice == "c":
        is_comparison = True

    return is_density, is_comparison


def density_query(
    encoder: str, vlmap: VLMap, text: str, complement: str, gs: int, names: np.ndarray, labels: np.ndarray
):
    svm_config = "/home/user/code/vlmaps/query_distribution/methods/config/svm_default.yaml"
    # svm_config = "/home/user/code/vlmaps/query_distribution/methods/config/cosine_svm_default.yaml"
    # svm_config = "/home/user/code/vlmaps/query_distribution/methods/config/old/old_svm_lseg_optimized.yaml"
    # svm_config = ""
    if encoder.lower() == "lseg":
        encoder_type = EncoderType.LSEG
    else:
        encoder_type = EncoderType.OPENSEG
    image_embedder, text_embedder = get_default_embedders(encoder_type, "/home/user/<path/to/quash>/temp")

    # print("0.  svm")
    # print("1.  cosine-svm")
    # print("2.  one-svm")
    # print("3.  fast-svm")
    # print("4.  dbscan")
    # print("5.  gp")
    # print("6.  knn")
    # print("7.  log")
    # print("8.  bayes")
    # print("9.  nn")
    # print("10. variance")
    # print("11. baseline")
    # print("12. normal")
    # classifier = "cosine-svm"
    # try:
    #     choice = int(input(": "))
    #     if choice == 0:
    #         classifier = "svm"
    #     elif choice == 1:
    #         classifier = "cosine-svm"
    #     elif choice == 2:
    #         classifier = "one-svm"
    #     elif choice == 3:
    #         classifier = "fast-svm"
    #     elif choice == 4:
    #         classifier = "dbscan"
    #     elif choice == 5:
    #         classifier = "gp"
    #     elif choice == 6:
    #         classifier = "knn"
    #     elif choice == 7:
    #         classifier = "log"
    #     elif choice == 8:
    #         classifier = "bayes"
    #     elif choice == 9:
    #         classifier = "nn"
    #     elif choice == 10:
    #         classifier = "variance"
    #     elif choice == 11:
    #         classifier = "baseline"
    #     elif choice == 12:
    #         classifier = "normal"
    # except:
    #     pass
    classifier = "cosine-svm"

    method = Method(
        TaskType.SEGMENTATION,
        encoder=encoder_type,
        verbose=True,
        classifier=classifier,
        classifier_config_path=svm_config,
        synonym_type=SynonymType.GENERATED,
        antonym_type=SynonymType.GENERATED,
        # synonym_type=SynonymType.NONE,
        # antonym_type=SynonymType.NONE,
        image_queries=False
    )
    # method = Method(TaskType.SEGMENTATION, encoder=encoder_type, verbose=True)

    # NN query
    if classifier == "nn":
        if text in names:
            res = np.where(names == text)
            i = res[0].item()
            label = int(labels[i])
            clf = method.get_get_predictor()
            preds = clf.predict(vlmap.grid_feat)
            mask = preds == label
            return mask

    # ch = input("Prompt engineering? y/n: ")
    # if ch == "y":
    #     method.prompt_engineering = PromptEngineering.MEAN
    # else:
    #     method.prompt_engineering = PromptEngineering.NONE
    method.prompt_engineering = PromptEngineering(0)

    # ch = input("Noise? y/n: ")
    # if ch == "y":
    #     print("1. synonym noise")
    #     print("2. antonym noise")
    #     print("3. both")
    #     noise_type = 0
    #     try:
    #         noise_type = int(input(": "))
    #         if noise_type == 1 or noise_type == 3:
    #             method.synonym_noise = True
    #         elif noise_type == 2 or noise_type == 3:
    #             method.antonym_noise = True
    #     except:
    #         print("Failure")

    #     try:
    #         noise_level = float(input("Min cosine sim: "))
    #         method.cosine_limit = noise_level
    #     except:
    #         print("Failure")

    #     try:
    #         noise_count = int(input("Noise count: "))
    #         method.noise_count = noise_count
    #     except:
    #         print("Failure")

    #     try:
    #         noise_type = int(input("Noise type: "))
    #         method.noise_type = noise_type
    #     except:
    #         print("Failure")

    postprocessing = False
    # ch = input("Postprocessing? y/n: ")
    # if ch == "y":
    #     postprocessing = True

    method.set_embedders(image_embedder, text_embedder)
    return run_method(method, vlmap, text, complement, postprocessing, gs)


def run_method(method: Method, vlmap: VLMap, text: str, complement: str, postprocessing: bool, gs: int):
    # print("Use synonyms?")
    # print("0/n: no")
    # print("1/y: synonyms and antonyms")
    # print("2/s: synonyms only")
    # print("3/a: antonyms only")
    # ch = input(": ")
    # synonyms, complement_synonyms = method.get_synonyms(text)
    # if ch == "2" or ch == "s":
    #     complement_synonyms = [complement]
    # elif ch == "3" or ch == "a":
    #     synonyms = [text]
    # elif ch == "0" or ch == "n":
    #     synonyms = [text]
    #     complement_synonyms = [complement]
    synonyms, complement_synonyms = method.get_synonyms(text, [complement])

    # if ch != "0" and ch != "n":
    #     try:
    #         ch = input("How many synonyms? -1/0 = all: ")
    #         num = int(ch)
    #         if num > 0 and num < len(synonyms):
    #             synonyms.remove(text)
    #             synonyms = random.sample(synonyms, num)
    #             synonyms.append(text)

    #         ch = input("How many antonyms? -1/0 = all: ")
    #         num = int(ch)
    #         if num > 0 and num < len(complement_synonyms):
    #             complement_synonyms.remove("other")
    #             if "background" in complement_synonyms:
    #                 complement_synonyms.remove("background")
    #             if "nothing" in complement_synonyms:
    #                 complement_synonyms.remove("nothing")
    #             complement_synonyms = random.sample(complement_synonyms, num)
    #             complement_synonyms.append("other")
    #             # complement_synonyms.append("background")
    #             # complement_synonyms.append("nothing")
    #     except:
    #         pass

    print("Synonyms:")
    print(synonyms)
    print("Complement synonyms:")
    print(complement_synonyms)
    mask = method.predict(
        vlmap.grid_feat, text, [complement], synonyms=synonyms, complement_synonyms=complement_synonyms
    )

    if postprocessing:
        mask = postprocess_3d(mask, vlmap.grid_pos, gs)

    return mask


def vlmaps_query(vlmap: VLMap, cat: str, gs: int):
    init_categories = False
    # use_prompt_engineering = True
    use_prompt_engineering = False
    index_2d = False
    postprocessing = False

    # ch = input("Init categories? y/n: ")
    # if ch == "y":
    #     init_categories = True
    # ch = input("Prompt engineering? y/n: ")
    # if ch == "y":
    #     use_prompt_engineering = True
    # ch = input("2D? y/n: ")
    # if ch == "y":
    #     index_2d = True
    # ch = input("Postprocessing? y/n: ")
    # if ch == "y":
    #     postprocessing = True

    if init_categories:
        print("considering categories: ")
        print(mp3dcat[1:-1])
        vlmap.init_categories(mp3dcat[1:-1], use_multiple_templates=use_prompt_engineering)
        mask, scores = vlmap.index_map(cat, use_prompt_engineering, with_init_cat=True)
    else:
        mask, scores = vlmap.index_map(cat, use_prompt_engineering, with_init_cat=False, add_other=True)
        print("scores from", np.min(scores), "to", np.max(scores))

    if index_2d:
        mask_2d = pool_3d_label_to_2d(mask, vlmap.grid_pos, gs)

        if postprocessing:
            foreground = binary_closing(mask_2d, iterations=3)
            foreground: np.ndarray = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
            foreground = foreground > 0.5
            foreground = binary_dilation(foreground)
            mask_2d = foreground
        return mask_2d, scores
    else:
        if postprocessing:
            mask = postprocess_3d(mask, vlmap.grid_pos, gs)

        return mask, scores


if __name__ == "__main__":
    main()
