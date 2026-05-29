import os
from typing import List, Dict, Tuple, Any
import random
import numpy as np
import json
from pycocotools.coco import COCO
from benchmarks.dataset import Dataset
from methods.common.tasktype import TaskType


class Coco(Dataset):
    def __init__(
        self,
        config_file: str,
        task: TaskType,
        shuffle: bool = False,
    ) -> None:
        super().__init__(config_file, task, shuffle, fetch_size=10, buffer_size=100)
        self.idx = 0
        self.ann_id = 0
        annotation_file: str = self.config.get_config_mandatory("annotation_file")
        self.img_dir: str = self.config.get_config_mandatory("img_dir")
        self.get_lists = self.config.get_config_mandatory("listmode", bool)

        self.category_names = {}
        category_ids = {}
        supercategories = {}
        names_list = []
        with open(annotation_file, "r", encoding="utf-8") as af:
            js = json.loads(af.read())
            categories_list = js["categories"]
            for cat in categories_list:
                id = cat["id"]
                name = cat["name"]
                supercategory = cat["supercategory"]
                self.category_names[id] = name
                category_ids[name] = id
                supercategories[id] = supercategory
                names_list.append(name)

        self.names_arr = np.array(names_list)

        # create dataset API
        self.coco: COCO = COCO(annotation_file=annotation_file)
        self.image_ids: List[int] = self.coco.getImgIds()

        if self.shuffle:
            random.shuffle(self.image_ids)

        self.ann_groups: Dict[str, List[dict]] = {}

        self.max_items = len(self.image_ids)

    def create_segmentation_item(self, overall_index) -> List[Tuple[Any, ...]]:
        ids = self.image_ids[overall_index]
        ids = [ids]
        img = self.coco.loadImgs(ids)
        assert self.img_dir
        image_path = self.img_dir + "/" + img[0]["file_name"]
        annotation_ids = self.coco.getAnnIds(imgIds=ids)
        init_anns = self.coco.loadAnns(annotation_ids)

        # download image
        exists = os.path.exists(image_path)
        if not exists:
            self.coco.download(tarDir=self.img_dir, imgIds=ids)

        ann_groups = {}
        for ann in init_anns:
            catid = ann["category_id"]
            if catid not in ann_groups:
                ann_groups[catid] = [ann]
            else:
                lst = ann_groups[catid]
                lst.append(ann)
                ann_groups[catid] = lst

        output = []
        for catid in ann_groups:
            anns = ann_groups[catid]
            catname = self.category_names[catid]
            gotmask = False
            gt_mask: np.ndarray = np.array(())  # type: ignore
            for ann in anns:
                temp_mask = self.coco.annToMask(ann)
                if not gotmask:
                    gotmask = True
                    gt_mask = temp_mask
                else:
                    gt_mask = gt_mask + temp_mask
            item = (catname, catid, gt_mask, image_path)
            output.append(item)
        return output

    def create_detection_item(self, overall_index) -> List[Tuple[Any, ...]]:
        ids = self.image_ids[overall_index]
        ids = [ids]
        img = self.coco.loadImgs(ids)
        assert self.img_dir
        image_path = self.img_dir + "/" + img[0]["file_name"]
        annotation_ids = self.coco.getAnnIds(imgIds=ids)
        init_anns = self.coco.loadAnns(annotation_ids)

        # download image
        exists = os.path.exists(image_path)
        if not exists:
            self.coco.download(tarDir=self.img_dir, imgIds=ids)

        ann_groups = {}
        for ann in init_anns:
            catid = ann["category_id"]
            if catid not in ann_groups:
                ann_groups[catid] = [ann]
            else:
                lst = ann_groups[catid]
                lst.append(ann)
                ann_groups[catid] = lst

        output = []
        for catid in ann_groups:
            anns = ann_groups[catid]
            catname = self.category_names[catid]
            bboxes = []
            for ann in anns:
                bbox = ann["bbox"]
                bboxes.append(bbox)
            item = (catname, catid, bboxes, image_path)
            output.append(item)
        return output

    def _create_item(self, overall_index) -> List[Tuple[Any, ...]]:
        # FIXME: this should be parametrized, not configured
        if self.task == TaskType.SEGMENTATION:
            return self.create_segmentation_item(overall_index)
        elif self.task == TaskType.DETECTION:
            return self.create_detection_item(overall_index)
        else:
            return []

    def get_categories(self) -> Dict[int, str]:
        return self.category_names

    def get_progress_str(self) -> str:
        progress_desc = "Img: "
        progress_desc += f"{self.idx:5d}"
        progress_desc += " / "
        progress_desc += f"{self.max_items:5d}"
        progress_desc += " Ann: "
        progress_desc += f"{self.ann_id:2d}"
        progress_desc += " / "
        progress_desc += f"{len(self.ann_groups):2d}"
        return progress_desc
