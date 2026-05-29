from typing import List, Dict, Tuple, Any
from pathlib import Path
import random
import numpy as np
from benchmarks.dataset import Dataset
from methods.common.tasktype import TaskType
import pandas as pd


class Matterport(Dataset):
    def __init__(
        self,
        config_file: str,
        task: TaskType,
        shuffle: bool = False,
    ) -> None:
        super().__init__(config_file, task, shuffle, fetch_size=10, buffer_size=100)
        self.idx = 0
        self.base_dir: str = self.config.get_config_mandatory("base_dir")
        self.category_path: str = self.config.get_config_mandatory("category_path")
        self.single_map = self.config.get_config_mandatory("single_map", bool)
        self.max_items = 0

        self.id_to_name, self.name_to_id = self.get_cats(self.category_path)
        self.data = self.get_imgs(self.base_dir)

        if self.shuffle:
            random.shuffle(self.data)

        self.max_items = len(self.data)
        self.ignored = ["void", "misc", "unlabeled"]

    def get_imgs(self, base_dir: str) -> List[Tuple[Path, Path]]:
        imgs = []
        labels = []
        path = Path(base_dir)
        num_maps = 0
        for map_dir in path.iterdir():
            if num_maps > 0 and self.single_map:
                break
            if str(map_dir.name).startswith("."):
                continue
            img_dir = f"{map_dir}/rgb-rt"
            label_dir = f"{map_dir}/semantic-rt"
            if Path(img_dir).exists() and Path(label_dir).exists():
                for img in Path(img_dir).iterdir():
                    imgs.append(img)
                for label in Path(label_dir).iterdir():
                    labels.append(label)
                num_maps += 1

        imgs = sorted(imgs)
        labels = sorted(labels)
        data = []

        for i, im in enumerate(imgs):
            label = labels[i]
            l_comp = str(label).replace("semantic", "rgb").replace("npy", "png")
            assert str(im) == l_comp
            data.append((im, label))

        return data

    def create_single_item(
        self, data_object: Tuple[Path, Path], id_to_name: Dict[int, str]
    ) -> List[Tuple[str, int, np.ndarray, str]]:
        (image_path, label_path) = data_object
        semantic_img = np.load(label_path)
        labels = np.unique(semantic_img)
        items = []
        for label in labels:
            gt_mask = semantic_img == label
            catname = id_to_name[label]
            if catname in self.ignored:
                continue
            item = (catname, label, gt_mask, str(image_path))
            items.append(item)
        return items

    def get_cats(self, path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
        df = pd.read_csv(path, delimiter="\t")
        labels = pd.unique(df["mpcat40index"])
        labels.sort()  # type: ignore
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

    def _create_item(self, overall_index) -> List[Tuple[Any, ...]]:
        return self.create_single_item(self.data[overall_index], self.id_to_name)

    def get_categories(self) -> Dict[int, str]:
        return self.id_to_name

    def get_progress_str(self) -> str:
        progress_desc = "Img: "
        progress_desc += f"{self.idx:5d}"
        progress_desc += " / "
        progress_desc += f"{self.max_items:5d}"
        return progress_desc
