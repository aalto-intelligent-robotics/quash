from benchmarks.dataset import Dataset
from benchmarks.coco import Coco
from benchmarks.coco_absence import CocoAbsence
from benchmarks.pascal_context import PascalContext
from benchmarks.ade_150 import ADE150
from benchmarks.ade_full import ADEFull
from benchmarks.matterport import Matterport
from methods.common.tasktype import TaskType


class Factory:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_dataset(
        dataset: str, config_file: str, task: TaskType, shuffle: bool = False
    ) -> Dataset:
        if dataset == "coco":
            return Coco(config_file, task, shuffle)
        elif dataset == "coco_absence":
            return CocoAbsence(config_file, task)
        elif (
            dataset == "pascal_context_59"
            or dataset == "pascal_context_459"
            or dataset == "pascal_context_all"
        ):
            return PascalContext(config_file, task)
        elif dataset == "ade_150":
            return ADE150(config_file, task)
        elif dataset == "ade_full" or dataset == "ade_all":
            return ADEFull(config_file, task)
        elif dataset == "matterport":
            return Matterport(config_file, task, shuffle)
        else:
            raise Exception("Unknown dataset")
