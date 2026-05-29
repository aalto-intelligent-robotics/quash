from typing import List, Tuple, Dict
from pathlib import Path
import numpy as np
import pandas as pd


def get_imgs(base_dir: str) -> List[Tuple[Path, Path]]:
    imgs = []
    labels = []
    path = Path(base_dir)
    for map_dir in path.iterdir():
        if str(map_dir.name).startswith("."):
            continue
        img_dir = f"{map_dir}/rgb-rt"
        for img in Path(img_dir).iterdir():
            imgs.append(img)
        label_dir = f"{map_dir}/semantic-rt"
        for label in Path(label_dir).iterdir():
            labels.append(label)

    imgs = sorted(imgs)
    labels = sorted(labels)
    data = []

    for i, im in enumerate(imgs):
        label = labels[i]
        l_comp = str(label).replace("semantic", "rgb").replace("npy", "png")
        assert str(im) == l_comp
        data.append((im, label))

    return data


def create_items(
    data_objects: List[Tuple[Path, Path]], id_to_name: Dict[int, str]
) -> List[List[Tuple[str, int, np.ndarray, str]]]:
    items = []
    for data_object in data_objects:
        items.append(create_item(data_object, id_to_name))
    return items


def create_item(
    data_object: Tuple[Path, Path], id_to_name: Dict[int, str]
) -> List[Tuple[str, int, np.ndarray, str]]:
    (image_path, label_path) = data_object
    semantic_img = np.load(label_path)
    labels = np.unique(semantic_img)
    items = []
    for label in labels:
        gt_mask = semantic_img == label
        catname = id_to_name[label]
        item = (catname, label, gt_mask, str(image_path))
        items.append(item)
    return items


def get_cats(path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    df = pd.read_csv(path, delimiter="\t")
    labels = pd.unique(df["mpcat40index"])
    labels.sort()
    id_to_name = {}
    name_to_id = {}

    for label in labels:
        mask = df["mpcat40index"] == label
        hits = df[mask]
        names = hits["mpcat40"]
        unames = pd.unique(names)
        assert len(unames) == 1
        name = unames[0]
        id_to_name[label] = name
        name_to_id[name] = label

    return id_to_name, name_to_id


def main() -> None:
    base_dir = "/home/user/data/datasets/vlmaps_dataset"
    category_path = "/home/user/code/vlmaps/cfg/mpcat40.tsv"
    id_to_name, name_to_id = get_cats(category_path)
    data = get_imgs(base_dir)
    items = create_item(data[0], id_to_name)

    for item in items:
        (catname, label, gt_mask, image_path) = item
        print(f"catname: {catname}")
        print(f"label: {label}")
        print(f"gt_mask: {gt_mask}")
        print(f"image_path: {image_path}")
        print("-" * 80)


if __name__ == "__main__":
    main()
