import random
import typing
from benchmarks.coco import Coco
import numpy as np
from methods.absence import AbsenceQuerier
from benchmarks.offline.offline_absence import absence_dictionary
from methods.common.tasktype import TaskType


class CocoAbsence(Coco):
    def __init__(self, config_file: str, task: TaskType) -> None:
        super().__init__(config_file, task)
        self.absence_queries = []
        model = self.config.get_config("absence_model")
        temp_folder = self.config.get_config("temp_folder")
        assert model
        assert temp_folder
        self.absence_querier = AbsenceQuerier(model, temp_folder)
        self.query_absence = self.config.get_config("query_absence", bool)
        self.offline = self.config.get_config("offline_absence", bool)

    def get_queries(self) -> typing.List[typing.Tuple[str, str]]:
        return self.absence_queries

    def get_mask(self) -> typing.Tuple[np.ndarray, str]:
        if self.offline:
            orig_catname = self.category_names[self.catid]
            catname = absence_dictionary[orig_catname]
        elif self.query_absence:
            orig_catname = self.category_names[self.catid]
            synonyms = self.absence_querier.get_absence_synonyms(orig_catname)
            catname = synonyms[0]
        else:
            no_match = self.names_arr
            for c in self.cats_in_img:
                orig_catname = self.category_names[c]
                no_match_mask = no_match != orig_catname
                no_match = no_match[no_match_mask]
            catname = random.choice(no_match)
        mask_size = self.coco.annToMask(self.anns[0])
        gt_mask = np.zeros_like(mask_size)
        self.absence_queries.append((orig_catname, catname))
        return gt_mask, catname
