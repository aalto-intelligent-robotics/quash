import os
import typing
from typing import List, Tuple, Any, Dict
import numpy as np
from PIL import Image
from benchmarks.dataset import Dataset
from methods.common.tasktype import TaskType


class ADE150(Dataset):
    def __init__(self, config_file: str, task: TaskType) -> None:
        super().__init__(config_file, task)
        self.data_dir = self.config.get_config("data_dir")
        assert self.data_dir

        self.set_used_classes()
        self.split = self.config.get_config("split")
        assert self.split

        info_file = self.data_dir + "/objectInfo150.txt"
        info = np.genfromtxt(info_file, delimiter="\t", dtype=str)
        self.idxs: np.ndarray = info[1:, 0].astype(dtype=np.int32)
        self.names: list[str] = list(info[1:, 4])

        self.imgs: list[str]
        self.annotations: list[str]
        if self.split == "both":
            t_imgs = self.getfiles("images/training")
            v_imgs = self.getfiles("images/validation")
            self.imgs = t_imgs + v_imgs
            t_imgs = self.getfiles("annotations/training")
            v_imgs = self.getfiles("annotations/validation")
            self.annotations = t_imgs + v_imgs
        else:
            self.imgs = self.getfiles("images/" + self.split)
            self.annotations = self.getfiles("annotations/" + self.split)

        self.max_items = len(self.imgs)

        self.name_dict: Dict[int, str] = {}
        for i in range(self.idxs.shape[0]):
            idx = self.idxs[i]
            name = self.names[i]
            self.name_dict[idx] = name

    def get_categories(self) -> typing.Dict[int, str]:
        return self.name_dict

    def get_name(self, id) -> str:
        idx_position = np.where(self.idxs == id)[0][0]
        return self.names[idx_position]

    def getfiles(self, path) -> typing.List:
        assert self.data_dir
        dir_path = os.path.join(self.data_dir, path)

        # Get the list of files in the directory and sort them by filename
        files = os.listdir(dir_path)
        files_sorted = sorted(files)  # Sort filenames alphabetically

        # Return the absolute paths of the sorted files
        return [os.path.abspath(os.path.join(dir_path, f)) for f in files_sorted]

    def _create_item(self, overall_index) -> List[Tuple[Any, ...]]:
        image_path = self.imgs[overall_index]
        ann_path = self.annotations[overall_index]
        img = Image.open(ann_path)  # Load the annotation
        ann = np.array(img)

        unique_values = np.unique(ann)

        output = []
        for id in unique_values:
            if id == 0:
                continue
            mask = (ann == id).astype(np.uint8)
            name = self.get_name(id)

            item = (name, id, mask, image_path)
            output.append(item)

        return output

    # TODO
    def get_progress_str(self) -> str:
        return ""
