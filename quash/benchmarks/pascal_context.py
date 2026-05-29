import os
import typing
from typing import List, Tuple, Any
import numpy as np
import scipy.io
from benchmarks.dataset import Dataset
from methods.common.tasktype import TaskType


class PascalContext(Dataset):
    def __init__(self, config_file: str, task: TaskType) -> None:
        super().__init__(config_file, task)
        self.img_dir = self.config.get_config_mandatory("img_dir", str)
        self.annotation_dir = self.config.get_config_mandatory("annotation_dir", str)
        self.annotations: list[str] = os.listdir(self.annotation_dir)
        self.set_used_classes()

        self.labels_file = self.config.get_config_mandatory("labels_file", str)

        self.label_names = {}
        with open(self.labels_file) as f:
            for line in f:
                (key, val) = line.split(": ")
                self.label_names[int(key)] = val.strip("\n")
        self.length = len(self.annotations)
        self.idx = 0
        self.label_idx = 0
        self.max_items = len(self.annotations)

    def _create_item(self, overall_index) -> List[Tuple[Any, ...]]:
        if overall_index >= len(self.annotations):
            return [()]
        annotation_file = self.annotations[overall_index]
        mat = scipy.io.loadmat(self.annotation_dir + "/" + annotation_file)
        image_path = self.img_dir + "/" + annotation_file.replace("mat", "jpg")
        arr = mat["LabelMap"]
        labels = np.unique(arr)
        output = []
        for label in labels:
            name = self.label_names[label]
            if not self.check_class(name):
                continue
            mask = arr == label
            item = (name, label, mask, image_path)
            output.append(item)
        return output

    def get_categories(self) -> typing.Dict[int, str]:
        return self.label_names

    def get_progress_str(self) -> str:
        progress_desc = "Img: "
        progress_desc += f"{self.idx:5d}"
        progress_desc += " / "
        progress_desc += f"{self.max_items:5d}"
        progress_desc += " Ann: "
        progress_desc += f"{self.label_idx:2d}"
        progress_desc += " / "
        progress_desc += f"{len(self.label_names):2d}"
        return progress_desc
