import torch
import json
from methods.method import Method
from methods.common.tasktype import TaskType


class NNMethod(torch.nn.Module, Method):
    def __init__(
        self,
        task: TaskType,
        num_of_synonyms: int = 21,
        environment: str = "home",
        classifier: str = "svm",
        api_model: str = "gpt4-turbo",
        measure: bool = False,
        logpath: str = "",
        temp_folder: str = "../temp/",
        max_retries: int = int(1e3),
    ) -> None:
        torch.nn.Module.__init__(self)
        Method.__init__(
            self,
            task,
            num_of_synonyms,
            environment,
            classifier,
            api_model,
            measure,
            logpath,
            temp_folder,
            max_retries,
        )
        self.init_embedders(
            weights_path="/home/user/<path/to/quash>/demo_e200.ckpt",
            device="cuda",
            network="clip",
            model_name="ViT-B/32",
            tempfolder="/home/user/<path/to/quash>/temp",
            cache=False,
        )
        self.training = False

    def setup(self, train_class_json: str, test_class_json: str):
        with open(train_class_json, "r") as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, "r") as f_in:
            self.test_class_texts = json.load(f_in)

    # Interface based on CAT-Seg
    def forward(self, x):
        dict = x[0]
        w = dict["width"]
        h = dict["height"]
        # img_T = dict["image"]
        # h = img_T.shape[1]
        # w = img_T.shape[2]

        y = torch.zeros((1, w, h))
        out = {}
        out["sem_seg"] = y
        return [out]
