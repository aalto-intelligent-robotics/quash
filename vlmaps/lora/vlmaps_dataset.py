import os
from typing import List, Tuple, Union, cast
import torch
from training_dataset import Dataset
import numpy as np
import pandas as pd
import open_clip
import torch.nn.functional as F
import pickle


class VLMapsDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        class_path: str,
        encoder: str,
        selected_maps: Union[List[int], List[str]],
        clip_model,
        tokenizer,
        device: str,
        write_mode: bool = False,
    ) -> None:
        if selected_maps and isinstance(selected_maps[0], int):
            map_ids = cast(List[int], selected_maps)
            map_strings = self.get_map_strings(map_ids)
        else:
            map_strings = cast(List[str], selected_maps)
        self.map_strings = map_strings
        self.data_path = data_path
        self.class_path = class_path
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.write_mode = write_mode
        self.device = device
        self.encoder = encoder
        if self.encoder != "lseg" and self.encoder != "openseg":
            raise ValueError(f"Unknown encoder {encoder}")

        if write_mode:
            self.load_complete_maps(data_path, map_strings)
            return

        self.indices, self.classes, self.class_embeddings = self.load_classes(
            class_path
        )
        self.semantics, self.embeddings = self.load_maps(data_path, map_strings)
        self.targets = self.class_embeddings[self.semantics]

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        return self.embeddings[idx].reshape(1, -1), self.targets[idx].squeeze()

    def get_description(self) -> str:
        out = ""
        out += f"self.data_path: {self.data_path}\n"
        out += f"self.class_path: {self.class_path}\n"
        # out += f"self.clip_model: {self.clip_model}\n"
        # out += f"self.tokenizer: {self.tokenizer}\n"
        out += f"self.write_mode: {self.write_mode}\n"
        out += f"self.device: {self.device}\n"
        out += f"self.map_strings: {self.map_strings}\n"
        return out

    def load_classes(self, class_path: str) -> Tuple[List[int], List[str], np.ndarray]:
        df = pd.read_csv(class_path, delimiter="\t")
        indices = df["mpcat40index"].to_list()
        classes = df["mpcat40"].to_list()
        return indices, classes, self.embed(classes)

    def find_maps(self, data_path: str, selected_maps: List[str]) -> List[str]:
        maps = []
        for f in os.listdir(data_path):
            if f in selected_maps or not selected_maps:
                dir_path = os.path.join(data_path, f)
                if os.path.isdir(dir_path):
                    full_path = os.path.join(data_path, f, "vlmap", f"parsed-alldata-{self.encoder}")
                    if os.path.isdir(full_path):
                        maps.append(full_path)

        assert maps, "No maps selected"
        return maps

    def load_maps(
        self, data_path: str, selected_maps: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        maps = self.find_maps(data_path, selected_maps)
        semantics = []
        all_embeddings = []
        for map in maps:
            semantic = np.load(f"{map}/grid_semantic.npy").squeeze()
            embeddings = np.load(f"{map}/grid_feat.npy")

            indices = np.argsort(semantic)
            semantic = semantic[indices]
            embeddings = embeddings[indices]

            semantics.append(semantic)
            all_embeddings.append(embeddings)
        semantics_array = np.concatenate(semantics)
        all_embeddings_array = np.concatenate(all_embeddings)
        return semantics_array, all_embeddings_array

    def load_complete_maps(self, data_path: str, selected_maps: List[str]):
        maps = self.find_maps(data_path, selected_maps)

        self.complete_maps = []
        for map in maps:
            embeddings = np.load(f"{map}/grid_feat.npy")
            with open(f"{map}/grid_histogram.pickle", "rb") as f:
                histogram = pickle.load(f)
            self.complete_maps.append(
                (map, embeddings, histogram)
            )

    def get_map_string_ids(self) -> List[str]:
        map_strings = [
            "5LpN3gDmAk7_1",  # 0
            "gTV8FGcVJC9_1",  # 1
            "jh4fc5c5qoQ_1",  # 2
            "JmbYfDe2QKZ_1",  # 3
            "JmbYfDe2QKZ_2",  # 4
            "mJXqzFtmKg4_1",  # 5
            "ur6pFq6Qu1A_1",  # 6
            "UwV83HsGsw3_1",  # 7
            "Vt2qJdWjCF2_1",  # 8
            "YmJkqBEsHnH_1",  # 9
        ]
        return map_strings

    def get_map_strings(self, map_ids: List[int]) -> List[str]:
        map_strings = self.get_map_string_ids()
        maps: list[str] = []
        for m in map_ids:
            maps.append(map_strings[m])
        return maps

    def get_unused_maps(self) -> List[str]:
        map_strings = self.get_map_string_ids()
        unused = []
        for map in map_strings:
            if map not in self.map_strings:
                unused.append(map)
        return unused

    def word2idx(self, query: str) -> int:
        idx = self.classes.index(query)
        return self.indices[idx]

    def idx2word(self, idx: int) -> str:
        id = self.indices.index(idx)
        return self.classes[id]

    def get_embeddings(self, idx: int) -> np.ndarray:
        mask = (self.semantics == idx).squeeze()
        return self.embeddings[mask, :]

    def get_class_embedding(self, query: str) -> np.ndarray:
        idx = self.classes.index(query)
        return self.class_embeddings[idx, :]

    def embed(self, words: List[str]) -> np.ndarray:
        if self.tokenizer is None or self.clip_model is None:
            return np.array(words)

        tokens = self.tokenizer(words).to(self.device)
        word_embeddings = self.clip_model.encode_text(tokens)
        word_embeddings = F.normalize(word_embeddings, dim=-1).detach().cpu().numpy()
        return word_embeddings

    def get_maps(
        self,
    ) -> List[
        Tuple[
            str,
            np.ndarray,
            dict
        ]
    ]:
        return self.complete_maps

    def get_classes(self) -> List[str]:
        return self.classes


if __name__ == "__main__":
    backbone = "ViT-B-32"
    clip_model, _, _ = open_clip.create_model_and_transforms(
        backbone, pretrained="openai", force_quick_gelu=True
    )
    clip_model.eval().to("cuda")
    tokenizer = open_clip.get_tokenizer(backbone)
    dataset = VLMapsDataset(
        "/home/user/hdd/datasets/vlmaps_dataset",
        "/home/user/code/vlmaps/cfg/mpcat40.tsv",
        "openseg",
        [],
        clip_model,
        tokenizer,
        "cuda",
    )
