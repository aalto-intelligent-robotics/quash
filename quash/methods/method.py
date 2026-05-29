import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import re
from typing import Tuple, List, Any, Optional, Dict, Final
from enum import Enum
from itertools import product, combinations
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import pandas as pd
import random
from scipy.ndimage import label
from sklearn import svm
from sklearn.metrics.pairwise import (
    cosine_similarity,
)
try:
    import clip
    CLIP_AVAILABLE = True
except:
    clip = None
    CLIP_AVAILABLE = False
try:
    from transformers import (
        Blip2Processor,
        Blip2ForConditionalGeneration,
    )

    BLIP_AVAILABLE = True
except:
    Blip2Processor = None  # type: ignore
    Blip2ForConditionalGeneration = None  # type: ignore
    BLIP_AVAILABLE = False
from transformers.utils.quantization_config import BitsAndBytesConfig
from methods.prompt import load_prompts
from methods.prompt_engineering import (
    get_engineered_mean_embedding,
    get_engineered_embeddings,
)
from methods.base.base import BasePredictor
from methods.common.query_api import QueryAPI
from methods.common.logging import get_log_file
from methods.common.tasktype import TaskType
from methods.common.creators.visualEncoderFactory import EncoderType
from methods.predictors import Predictor, PredictorFactory


class PromptEngineering(Enum):
    NONE = 0
    MEAN = 1
    ALL = 2
    ONLY = 3


class SynonymType(Enum):
    GENERATED = 0
    LIST = 1
    LEXICON = 2
    ADJECTIVE = 3
    PERFECT = 4
    NONE = 5


# ███╗   ███╗███████╗████████╗██╗  ██╗ ██████╗ ██████╗
# ████╗ ████║██╔════╝╚══██╔══╝██║  ██║██╔═══██╗██╔══██╗
# ██╔████╔██║█████╗     ██║   ███████║██║   ██║██║  ██║
# ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║   ██║██║  ██║
# ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║╚██████╔╝██████╔╝
# ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═════╝


class Method(BasePredictor):
    def __init__(
        self,
        task: TaskType,
        encoder: EncoderType,
        num_of_synonyms: int = 21,
        environment: str = "home",
        classifier: str = "svm",
        api_model: str = "gpt4-turbo",
        synonym_type: SynonymType = SynonymType.GENERATED,
        antonym_type: SynonymType = SynonymType.GENERATED,
        antonym_ratio: int = 5,
        measure: bool = False,
        logpath: str = "./log",
        temp_folder: str = "",
        max_retries: int = int(1e3),
        predict_environment: bool = False,
        classifier_config_path: str = "",
        prompts: str = "default",
        prompt_engineering: PromptEngineering = PromptEngineering.NONE,
        verbose: bool = False,
        image_queries: bool = False,
    ) -> None:
        super().__init__(measure)

        # parametrization
        self.task = task
        self.encoder = encoder
        self.default_environment = environment
        self.num_of_synonyms = num_of_synonyms
        self.classifier = classifier
        self.api_model = api_model
        self.dummy_mode = self.classifier == "dummy"
        self.predict_environment = predict_environment
        self.classifier_config_path = classifier_config_path
        self.inject_predictor = False
        self.prompt_engineering = prompt_engineering
        self.verbose = verbose
        (
            self.synonym_prompt,
            self.antonym_prompt,
            self.synonym_system_prompt,
            self.antonym_system_prompt,
        ) = load_prompts(prompts)
        self.synonym_type = synonym_type
        self.antonym_type = antonym_type
        self.antonym_ratio = antonym_ratio
        self.image_queries = image_queries
        # vars
        self.query_cache: Dict[str, Tuple[List[str], List[str]]] = {}
        self.predictors: Dict[str, Predictor] = {}
        self.synonym_list = pd.DataFrame()
        self.lexicon = []
        self.adjective_list = pd.DataFrame()
        self.zero_vector_key: Final[str] = "<<zero>>"
        self.clip_image_encoder = None
        # logging
        self.logfile = get_log_file(logpath)
        # temp folder
        if not temp_folder:
            if Path("/home/user").exists():
                temp_folder = "/home/user/code/vlmaps/temp"
            elif Path("/home/user").exists():
                temp_folder = "/home/user/code/vlmaps/temp"
            else:
                temp_folder = "./temp"
        # API
        self.queryAPI = QueryAPI(
            model=self.api_model, temp_folder=temp_folder, max_retries=max_retries
        )
        self.synonym_noise = False
        self.antonym_noise = False
        self.cosine_limit = 0.0
        self.noise_count = 0
        self.noise_type = 1

    def get_method_description(self, short: bool = False) -> str:
        description = ""
        description += "Classifier: " + self.classifier + "\n"
        try:
            if self.predictors is not None and len(self.predictors) > 0:
                clf = next(iter(self.predictors.values()))
                description += "Classifier options: \n"
                description += clf.get_description()
        except Exception:
            pass
        description += "API model: " + self.api_model + "\n"
        description += "Environment: " + self.default_environment + "\n"
        description += "Num of synonyms: " + str(self.num_of_synonyms) + "\n"
        description += "Logfile: " + self.logfile + "\n"
        if not short:
            description += "\n"
            s_prompt, a_prompt = self.get_prompts("<query>")
            description += "Prompts: \n"
            description += "-" * 20 + "\n"
            description += self.synonym_system_prompt + "\n"
            description += "-" * 20 + "\n"
            description += self.antonym_system_prompt + "\n"
            description += "-" * 20 + "\n"
            description += s_prompt + "\n"
            description += "-" * 20 + "\n"
            description += a_prompt
        return description

    def get_parameters(self):
        return (
            self.default_environment,
            self.num_of_synonyms,
            self.classifier,
            self.api_model,
        )

    def get_prompts(
        self, query: str, environment: Optional[str] = None
    ) -> Tuple[str, str]:
        if environment:
            env = environment
        else:
            env = self.default_environment
        s_prompt = (
            self.synonym_prompt.replace("<query>", query)
            .replace("<env>", env)
            .replace("<num>", str(self.num_of_synonyms))
        )
        a_prompt = (
            self.antonym_prompt.replace("<query>", query)
            .replace("<env>", env)
            .replace("<num>", str(self.num_of_synonyms))
        )
        return s_prompt, a_prompt

    def parse_list_string(self, s: str) -> List[str]:
        # Normalize whitespace
        s = s.strip()

        # Try to detect numbered list: "1. A, 2. B" or "1. A\n2. B"
        numbered_pattern = r"(?:\d+\.\s*)([^\n,]+)"
        matches = re.findall(numbered_pattern, s)
        if matches:
            return [item.strip() for item in matches]

        # Try comma-separated list
        if "," in s:
            items = [item.strip() for item in s.split(",") if item.strip()]
            if len(items) > 1:
                return items

        # Try newline-separated list
        lines = [line.strip() for line in s.split("\n") if line.strip()]
        if len(lines) > 1:
            return lines

        # Fallback: single item
        return [s]

    def get_synonyms(
        self,
        query: str,
        complement: List[str],
        environment: Optional[str] = None,
        verbose: bool = False,
    ) -> Tuple[List[str], List[str]]:
        if self.synonym_type == SynonymType.GENERATED:
            synonyms = self.get_synonyms_query(query, environment, verbose)
        elif self.synonym_type == SynonymType.LIST:
            synonyms = self.get_synonyms_list(query)
        elif self.synonym_type == SynonymType.LEXICON:
            synonyms = self.get_lexicon_words(self.num_of_synonyms, complement=False)
        elif self.synonym_type == SynonymType.ADJECTIVE:
            synonyms = self.get_adjectives([query])
        elif self.synonym_type == SynonymType.PERFECT:
            synonyms = [query]
        elif self.synonym_type == SynonymType.NONE:
            synonyms = [query]
        else:
            raise ValueError("Unknown synonym type")

        if self.antonym_type == SynonymType.GENERATED:
            antonyms = self.get_antonyms_query(query, environment, verbose)
        elif self.antonym_type == SynonymType.LIST:
            antonyms = self.get_antonyms_list(query)
        elif self.antonym_type == SynonymType.LEXICON:
            antonyms = self.get_lexicon_words(self.antonym_ratio, complement=True)
        elif self.antonym_type == SynonymType.ADJECTIVE:
            antonyms = self.get_adjectives(complement)
        elif self.antonym_type == SynonymType.PERFECT:
            antonyms = self.get_perfect_antonyms(query)
        elif self.synonym_type == SynonymType.NONE:
            antonyms = complement
        else:
            raise ValueError("Unknown antonym type")
        self.query_cache[query] = (synonyms, antonyms)

        return synonyms, antonyms

    def get_synonyms_list(self, query: str) -> List[str]:
        # load mapping
        if self.synonym_list.empty:
            method_dir = os.environ.get("METHOD_BASE_DIR")
            assert method_dir, "source setup_env.sh"
            self.synonym_list = pd.read_csv(
                f"{method_dir}/methods/classes/category_mapping.csv",
                delimiter=";",
                index_col="index",
            )
        assert not self.synonym_list.empty

        # synonyms
        all_synonyms = self.synonym_list[self.synonym_list["mpcat40"] == query][
            "raw_category"
        ].to_list()
        num_pick_syn = min(self.num_of_synonyms, len(all_synonyms))
        synonyms = random.sample(all_synonyms, num_pick_syn)
        synonyms.append(query)

        return synonyms

    def get_antonyms_list(self, query: str) -> List[str]:
        # load mapping
        if self.synonym_list.empty:
            method_dir = os.environ.get("METHOD_BASE_DIR")
            assert method_dir, "source setup_env.sh"
            self.synonym_list = pd.read_csv(
                f"{method_dir}/methods/classes/category_mapping.csv",
                delimiter=";",
                index_col="index",
            )
        assert not self.synonym_list.empty

        # antonyms
        all_antonyms = self.synonym_list[self.synonym_list["mpcat40"] != query][
            "raw_category"
        ].to_list()
        # NOW
        # NOTE: important tuning parameter
        num_pick_ant = min(self.num_of_synonyms * self.antonym_ratio, len(all_antonyms))
        antonyms = random.sample(all_antonyms, num_pick_ant)
        antonyms.append("other")
        antonyms.append("background")
        antonyms.append("nothing")
        antonyms.append("room")
        antonyms.append("space")

        return antonyms

    def get_perfect_antonyms(self, query: str) -> List[str]:
        # load mapping
        if self.synonym_list.empty:
            method_dir = os.environ.get("METHOD_BASE_DIR")
            assert method_dir, "source setup_env.sh"
            self.synonym_list = pd.read_csv(
                f"{method_dir}/methods/classes/mpcat40.csv",
                delimiter=";",
                index_col="mpcat40index",
            )
        assert not self.synonym_list.empty

        # antonyms
        antonyms = self.synonym_list[self.synonym_list["mpcat40"] != query][
            "mpcat40"
        ].to_list()
        antonyms.append("other")
        # antonyms.append("background")
        # antonyms.append("nothing")
        # antonyms.append("room")
        # antonyms.append("space")

        return antonyms

    def sample_adjectives(self, noun: str, sample_counts: List[int]) -> List[str]:
        method_dir = os.environ.get("METHOD_BASE_DIR")
        assert method_dir, "source setup_env.sh"
        df = pd.read_csv(
            f"{method_dir}/methods/classes/adjectives.csv",
            delimiter=",",
        )

        results = []

        # Group adjectives by their type
        grouped = df.groupby("class")["adjective"].apply(list)
        all_types = list(grouped.index)

        for k, n_samples in enumerate(sample_counts):
            if k == 0:
                results.extend([noun] * n_samples)
                continue

            combos = []
            for type_combo in combinations(all_types, k):
                adjective_lists = [grouped[t] for t in type_combo]
                for adj_combo in product(*adjective_lists):
                    phrase = " ".join(adj_combo + (noun,))
                    combos.append(phrase)

            if n_samples < 0:
                n_samples = len(combos)

            sampled = random.sample(combos, min(n_samples, len(combos)))
            results.extend(sampled)

        return results

    def get_adjectives(self, query: List[str]) -> List[str]:
        # load mapping
        if self.adjective_list.empty:
            method_dir = os.environ.get("METHOD_BASE_DIR")
            assert method_dir, "source setup_env.sh"
            self.adjective_list = pd.read_csv(
                f"{method_dir}/methods/classes/adjectives.csv",
                delimiter=",",
            )
        assert not self.adjective_list.empty

        # synonyms
        synonyms = []
        for q in query:
            s = self.sample_adjectives(q, [1, -1, 10, 5])
            synonyms += s

        return synonyms

    def get_lexicon_words(self, ratio: int, complement: bool) -> List[str]:
        # load mapping
        if not self.lexicon:
            method_dir = os.environ.get("METHOD_BASE_DIR")
            assert method_dir, "source setup_env.sh"
            with open(f"{method_dir}/methods/classes/words_alpha.txt", "r") as f:
                lines = f.readlines()
                self.lexicon = [line.replace("\n", "") for line in lines]
        assert self.lexicon

        # antonyms
        num_pick_ant = min(self.num_of_synonyms * ratio, len(self.lexicon))
        antonyms = random.sample(self.lexicon, num_pick_ant)
        if complement:
            antonyms.append("other")
            antonyms.append("background")
            antonyms.append("nothing")

        return antonyms

    def get_synonyms_and_antonyms_query(
        self, query: str, environment: Optional[str] = None, verbose: bool = False
    ) -> Tuple[List[str], List[str]]:
        if self.dummy_mode:
            return ["dummy"], ["dummy"]

        if query in self.query_cache:
            return self.query_cache[query]

        s_prompt, a_prompt = self.get_prompts(query, environment)

        got_synonyms = False
        got_antonyms = False
        synonyms: Optional[List[str]] = None
        antonyms: Optional[List[str]] = None

        # get synonyms
        if not got_synonyms:
            synonyms_raw = self.queryAPI.query(
                self.synonym_system_prompt, s_prompt, verbose
            )
            # synonyms = synonyms_raw.split(",")
            synonyms = self.parse_list_string(synonyms_raw)
            for i, synonym in enumerate(synonyms):
                synonyms[i] = synonym.strip()

        # get antonyms
        if not got_antonyms:
            a_prompt = a_prompt.replace("<synonyms>", str(synonyms))
            antonyms_raw = self.queryAPI.query(
                self.antonym_system_prompt, a_prompt, verbose
            )
            # antonyms = antonyms_raw.split(",")
            antonyms = self.parse_list_string(antonyms_raw)
            for i, antonym in enumerate(antonyms):
                antonyms[i] = antonym.strip()

        assert synonyms, "Synonyms missing"
        assert antonyms, "Antonyms missing"

        synonyms.append(query)
        antonyms.append("other")
        antonyms.append("background")
        antonyms.append("nothing")

        self.query_cache[query] = (synonyms, antonyms)

        with open(self.logfile, "a", encoding="utf-8") as f:
            string = query + " synonyms:"
            for s in synonyms:
                string += s + ","
            string += "\n"
            f.write(string)
            string = query + " antonyms:"
            for s in antonyms:
                string += s + ","
            string += "\n"
            f.write(string)
            f.close()

        return synonyms, antonyms

    def get_synonyms_query(
        self, query: str, environment: Optional[str] = None, verbose: bool = False
    ) -> List[str]:
        if self.dummy_mode:
            return ["dummy"]

        if query in self.query_cache:
            synonyms, _ = self.query_cache[query]
            return synonyms

        s_prompt, a_prompt = self.get_prompts(query, environment)

        got_synonyms = False
        synonyms: Optional[List[str]] = None

        # get synonyms
        if not got_synonyms:
            synonyms_raw = self.queryAPI.query(
                self.synonym_system_prompt, s_prompt, verbose
            )
            # synonyms = synonyms_raw.split(",")
            synonyms = self.parse_list_string(synonyms_raw)
            for i, synonym in enumerate(synonyms):
                synonyms[i] = synonym.strip()

        assert synonyms, "Synonyms missing"

        synonyms.append(query)

        with open(self.logfile, "a", encoding="utf-8") as f:
            string = query + " synonyms:"
            for s in synonyms:
                string += s + ","
            string += "\n"
            f.write(string)
            f.close()

        return synonyms

    def get_antonyms_query(
        self, query: str, environment: Optional[str] = None, verbose: bool = False
    ) -> List[str]:
        if self.dummy_mode:
            return ["dummy"]

        if query in self.query_cache:
            _, antonyms = self.query_cache[query]
            return antonyms

        s_prompt, a_prompt = self.get_prompts(query, environment)

        got_antonyms = False
        antonyms: Optional[List[str]] = None

        # get antonyms
        if not got_antonyms:
            antonyms_raw = self.queryAPI.query(
                self.antonym_system_prompt, a_prompt, verbose
            )
            # antonyms = antonyms_raw.split(",")
            antonyms = self.parse_list_string(antonyms_raw)
            for i, antonym in enumerate(antonyms):
                antonyms[i] = antonym.strip()

        assert antonyms, "Antonyms missing"

        antonyms.append("other")
        antonyms.append("background")
        antonyms.append("nothing")

        with open(self.logfile, "a", encoding="utf-8") as f:
            string = query + " antonyms:"
            for s in antonyms:
                string += s + ","
            string += "\n"
            f.write(string)
            f.close()

        return antonyms

    def upscale(self, mask: np.ndarray, width: int, height: int):
        mask_width = mask.shape[1]
        mask_height = mask.shape[0]

        # correct shape, do nothing
        if mask_width == width and mask_height == height:
            return mask

        # fix the size
        # Convert the numpy array to a PIL image
        # (keep it binary by using 'L' mode for grayscale and scale values)
        pil_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")

        # Resize using Pillow
        resized_pil_mask = pil_mask.resize(
            (width, height), Image.Resampling.NEAREST
        )  # NEAREST to preserve binary nature

        # Convert back to numpy array and threshold to get binary values
        resized_mask = np.array(resized_pil_mask)
        resized_mask = (resized_mask > 128).astype(
            np.uint8
        )  # Threshold to convert to binary (0 and 1)

        return resized_mask

    def dummy_prediction(self, image_path: str) -> Tuple[np.ndarray, List[List[int]]]:
        _, width, height, _, _ = self.image_embedder.read(image_path)
        mask = np.zeros((height, width))

        up = int(height / 4)
        down = int(height * 3 / 4)
        left = int(width / 4)
        right = int(width * 3 / 4)
        bboxes = []
        bbox = [left, down, right, up]
        bboxes.append(bbox)

        return mask, bboxes

    def predict_and_load_image_batch(
        self, items: List[Tuple[Any, ...]], complement: List[str]
    ) -> list:
        _, _, _, image_path = items[0]

        if self.dummy_mode:
            return [self.dummy_prediction(image_path)]

        image_embeddings, _, width, height = self.image_embedder.read_and_embed(
            image_path
        )

        output = []

        for item in items:
            query, _, _, image_path = item
            mask = self._image_prediction(image_embeddings, query, complement)

            mask = self.upscale(mask, width, height)
            bboxes = self.fit_bounding_box(mask)

            output.append((mask, bboxes))

        return output

    def predict_and_load_image(
        self,
        image_path: str,
        query: str,
        complement: List[str],
        environment: Optional[str] = None,
        synonyms: Optional[List[str]] = None,
        complement_synonyms: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[List[int]], np.ndarray, np.ndarray, np.ndarray]:
        if self.dummy_mode:
            mask, bboxes = self.dummy_prediction(image_path)
            subtraction = np.zeros_like(mask)
            distances = np.zeros_like(mask)
            complement_distances = np.zeros_like(mask)
            return mask, bboxes, subtraction, distances, complement_distances

        if self.measure:
            self.start_timer("method")

        image_embeddings, image, width, height = self.image_embedder.read_and_embed(
            image_path
        )
        self.image = image

        if self.predict_environment:
            env, success = self.describe_environment(image)
            if not success:
                env = self.default_environment
        else:
            if environment:
                env = environment
            else:
                env = self.default_environment

        mask = self._image_prediction(
            image_embeddings,
            query,
            complement,
            env,
            synonyms,
            complement_synonyms,
        )

        mask = self.upscale(mask, width, height)
        bboxes = self.fit_bounding_box(mask)

        if self.measure:
            self.end_timer()

        subtraction = np.zeros_like(mask)
        distances = np.zeros_like(mask)
        complement_distances = np.zeros_like(mask)
        return mask, bboxes, subtraction, distances, complement_distances

    def describe_environment(self, image: np.ndarray) -> Tuple[str, bool]:
        if not BLIP_AVAILABLE:
            return "", False

        assert Blip2Processor is not None
        assert Blip2ForConditionalGeneration is not None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "Salesforce/blip2-opt-2.7b"
        dtype = torch.float32

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_use_double_quant=True,
        )
        processor = Blip2Processor.from_pretrained(
            model_name, use_fast=False, torch_dtype=dtype
        )
        assert isinstance(processor, Blip2Processor)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=dtype,
        )  # doctest: +IGNORE_RESULT

        prompt = ""
        if prompt:
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                device, dtype  # type: ignore
            )
        else:
            inputs = processor(images=image, return_tensors="pt").to(
                device, dtype  # type: ignore
            )

        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # FIXME: remove verbosity
        print("Predicted environment")
        print(generated_text)

        return generated_text, True

    def fit_bounding_box(self, mask: np.ndarray):
        # Find the non-zero (white) pixel coordinates
        labeled_image, num_features = label(mask > 0)  # type: ignore

        bboxes = []
        for label_id in range(1, num_features + 1):
            # Find the coordinates of the pixels that belong to this component
            component_coords = np.argwhere(labeled_image == label_id)

            if component_coords.size > 0:
                y_min, x_min = component_coords.min(axis=0)
                y_max, x_max = component_coords.max(axis=0)

                bbox = [x_min, y_min, x_max, y_max]
                bboxes.append(bbox)
        return bboxes

    def predict_image(
        self,
        image: np.ndarray,
        filename: str,
        query: str,
        complement: List[str],
        synonyms: list,
        complement_synonyms: list,
        environment: Optional[str] = None,
        base_size: int = -1,
        crop_size: int = -1,
    ) -> np.ndarray:
        if self.measure:
            self.start_timer("method")

        image_embeddings, _ = self.image_embedder.embed(
            image, filename, base_size, crop_size
        )

        mask = self._image_prediction(
            image_embeddings,
            query,
            complement,
            environment,
            synonyms,
            complement_synonyms,
        )

        if self.measure:
            self.end_timer()

        return mask

    def _image_prediction(
        self,
        image_embeddings: np.ndarray,
        query: str,
        complement: List[str],
        environment: Optional[str] = None,
        synonyms: Optional[List[str]] = None,
        complement_synonyms: Optional[List[str]] = None,
    ) -> np.ndarray:
        if self.measure:
            self.start_timer("method")

        height = image_embeddings.shape[0]
        width = image_embeddings.shape[1]
        depth = image_embeddings.shape[2]
        image_embeddings = image_embeddings.reshape(-1, depth)

        mask = self.predict(
            image_embeddings,
            query,
            complement,
            environment,
            synonyms,
            complement_synonyms,
        )

        mask = mask.reshape((height, width))

        return mask

    def got_predictor(self, query) -> bool:
        return query in self.predictors

    def set_get_predictor(self, f) -> None:
        self.inject_predictor = True
        self.injected_predictor = f

    def get_get_predictor(self) -> Predictor:
        if not self.inject_predictor:
            return PredictorFactory.get_predictor(
                self.classifier, self.classifier_config_path, self.encoder
            )
        else:
            return self.injected_predictor

    def get_embeddings(self, queries: List[str], synonyms: List[str]) -> np.ndarray:
        synonym_embeddings_l = []
        if self.prompt_engineering == PromptEngineering.ONLY:
            for query in queries:
                synonym_embeddings_l.append(
                    get_engineered_embeddings(query, self.text_embedder.embed)
                )
        else:
            for s in synonyms:
                if self.prompt_engineering == PromptEngineering.MEAN:
                    synonym_embeddings_l.append(
                        get_engineered_mean_embedding(s, self.text_embedder.embed)
                    )
                elif self.prompt_engineering == PromptEngineering.ALL:
                    synonym_embeddings_l.append(
                        get_engineered_embeddings(s, self.text_embedder.embed)
                    )
                else:
                    synonym_embeddings_l.append(self.text_embedder.embed(s))
        synonym_embeddings = np.array(synonym_embeddings_l)
        return synonym_embeddings

    def create_predictor(
        self,
        query: str,
        complement: List[str],
        environment: Optional[str] = None,
        synonyms: Optional[List[str]] = None,
        complement_synonyms: Optional[List[str]] = None,
    ):
        if self.image_queries:
            (
                synonym_embeddings,
                complement_synonym_embeddings,
                query_embedding,
                complement_embeddings,
            ) = self.get_image_synonyms(query)
        else:
            if synonyms is None or complement_synonyms is None:
                synonyms, complement_synonyms = self.get_synonyms(
                    query, complement, environment
                )

            # embed synonyms
            if query not in synonyms:
                synonyms.append(query)
            synonym_embeddings = self.get_embeddings([query], synonyms)
            query_index = synonyms.index(query)
            query_embedding = synonym_embeddings[query_index, :]

            # embed antonyms
            # special case: zero embedding
            if len(complement) == 1 and complement[0] == self.zero_vector_key:
                complement_embeddings = np.zeros_like(query_embedding)
                complement_synonym_embeddings = complement_embeddings.copy().reshape(
                    1, -1
                )
            else:
                for comp in complement:
                    if comp not in complement_synonyms:
                        complement_synonyms.append(comp)
                complement_synonym_embeddings = self.get_embeddings(
                    complement, complement_synonyms
                )
                complement_indices = [
                    complement_synonyms.index(q)
                    for q in complement
                    if q in complement_synonyms
                ]
                complement_embeddings = complement_synonym_embeddings[
                    complement_indices, :
                ]

        noise = self.synonym_noise or self.antonym_noise
        if noise:
            if self.synonym_noise:
                synonym_embeddings = self.add_noise(synonym_embeddings)
            if self.antonym_noise:
                complement_synonym_embeddings = self.add_noise(
                    complement_synonym_embeddings
                )

        X = np.concatenate((synonym_embeddings, complement_synonym_embeddings), axis=0)
        y = np.concatenate(
            (
                np.ones(synonym_embeddings.shape[0]),
                np.zeros(complement_synonym_embeddings.shape[0]),
            )
        )
        # clf = PredictorFactory.get_predictor(
        #     self.classifier, self.classifier_config_path
        # )
        clf = self.get_get_predictor()
        # note: only in parameter tuning this is the case - this is SVC
        if isinstance(clf, Predictor):
            clf.set_query(query, query_embedding)
            clf.set_complement(complement, complement_embeddings)
        clf.fit(X, y)
        self.predictors[query] = clf

        if self.verbose:
            print(f"Predictor: {self.classifier}")
            print(f"{self.classifier_config_path}")
            print(clf.get_description())

    def predict(
        self,
        embeddings: np.ndarray,
        query: str,
        complement: List[str],
        environment: Optional[str] = None,
        synonyms: Optional[List[str]] = None,
        complement_synonyms: Optional[List[str]] = None,
    ) -> np.ndarray:
        if not self.got_predictor(query):
            self.create_predictor(
                query, complement, environment, synonyms, complement_synonyms
            )
        clf: Predictor = self.predictors[query]

        mask = clf.predict(embeddings)
        return mask

    def classify(
        self,
        embeddings: torch.Tensor,
        synonym_embeddings: torch.Tensor,
        complement_synonym_embeddings: torch.Tensor,
        dim1: int,
        dim2: int,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        word_embeddings = synonym_embeddings + complement_synonym_embeddings
        X = torch.cat([s for s in word_embeddings], dim=0).detach().cpu().numpy()
        y = (
            torch.cat(
                (
                    torch.ones(len(synonym_embeddings)),
                    torch.zeros(len(complement_synonym_embeddings)),
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
        clf = svm.SVC()
        clf.fit(X, y)

        cpu_embeddings = embeddings.detach().cpu().numpy()
        length = embeddings.shape[0]
        cpu_mask = np.zeros((length))
        for i in tqdm(range(length), leave=False, desc="Method classify"):  # type: ignore
            embedding = cpu_embeddings[i, :]
            embedding = embedding.reshape((1, -1))
            cpu_mask[i] = clf.predict(embedding)

        mask = torch.Tensor(cpu_mask).to(torch.device("cuda"))
        mask = mask.float().view(1, dim1, dim2, -1).permute(0, 3, 1, 2)

        return mask, cpu_mask.reshape((dim1, dim2))

    def get_num_queries(self) -> Tuple[int, int]:
        return self.queryAPI.get_num_queries()

    def perturb_vector(
        self, v0: np.ndarray, min_cosine_sim: float
    ) -> Optional[np.ndarray]:
        """
        Rotates the unit vector v0 by a random angle up to max_angle_rad
        in a random orthogonal direction.
        """
        max_it = 1000
        it = 0
        while it < max_it:
            max_angle_rad = np.arccos(min_cosine_sim)
            d = len(v0)
            v0 = v0 / np.linalg.norm(v0)

            # Step 1: Generate random orthogonal direction
            rand = np.random.randn(d)
            proj = rand - np.dot(rand, v0) * v0  # make orthogonal to v0
            ortho = proj / np.linalg.norm(proj)

            # Step 2: Choose random rotation angle θ in [0, max_angle_rad]
            theta = np.random.uniform(0, max_angle_rad)

            # Step 3: Construct rotated vector
            rotated = np.cos(theta) * v0 + np.sin(theta) * ortho
            if min_cosine_sim <= cosine_similarity(
                rotated.reshape(1, -1), v0.reshape(1, -1)
            ):
                return rotated
        return None

    def get_noise_vectors(
        self, v0: np.ndarray, max_cosine_sim: float, count: int
    ) -> np.ndarray:
        vecs = []
        for i in range(count):
            noise_vec = self.perturb_vector(v0, max_cosine_sim)
            if noise_vec is not None:
                vecs.append(noise_vec)
        return np.stack(vecs, axis=1)

    def add_noise(self, vectors: np.ndarray, override: bool = False) -> np.ndarray:
        if self.noise_type == 1:
            return self.add_sample_noise(vectors, override)
        else:
            return self.add_mean_noise(vectors, override)

    def add_mean_noise(self, vectors: np.ndarray, override: bool) -> np.ndarray:
        vec = np.mean(vectors, axis=0)
        noise = (
            self.get_noise_vectors(vec, self.cosine_limit, self.noise_count)
            .transpose()
            .astype(dtype=np.float32)
        )
        cosines = cosine_similarity(noise, vec.reshape(1, -1)).flatten()
        all_ok = np.all(cosines > self.cosine_limit)
        if not all_ok:
            print("All not ok!")

        out = np.concatenate((vectors, noise), axis=0)
        return out

    def add_sample_noise(self, vectors: np.ndarray, override: bool) -> np.ndarray:
        embeddings = [vectors]
        for i in range(vectors.shape[0]):
            vec = vectors[i, :]
            noise = (
                self.get_noise_vectors(vec, self.cosine_limit, self.noise_count)
                .transpose()
                .astype(dtype=np.float32)
            )
            cosines = cosine_similarity(noise, vec.reshape(1, -1)).flatten()
            all_ok = np.all(cosines > self.cosine_limit)
            if not all_ok:
                print("All not ok!")

            embeddings.append(noise)
        out = np.concatenate(embeddings, axis=0)
        return out

    def match_filter(self, string: str, filter: str = "") -> bool:
        if not filter:
            return True
        match = re.search(filter, string)
        return match is not None

    def encode_image(self, file_path: str) -> np.ndarray:
        assert CLIP_AVAILABLE
        if self.clip_image_encoder is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_image_encoder, self.clip_image_encoder_preprocess = clip.load("ViT-B/32", device=self.device)

        image = self.clip_image_encoder_preprocess(Image.open(file_path)).unsqueeze(0).to(self.device)
        image_features = self.clip_image_encoder.encode_image(image).detach().cpu().numpy()
        return image_features

    def get_image_synonyms(
        self,
        query: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        path = os.environ.get("IMAGE_SYNONYM_PATH")
        assert path, "source setup_environment.sh"

        files = [
            f
            for f in listdir(path)
            if isfile(join(path, f))
        ]

        synonym_files = []
        other_files = []
        for file in files:
            if self.match_filter(file, f"{query}.+"):
                synonym_files.append(file)
            else:
                other_files.append(file)

        embeddings = []
        for file in synonym_files:
            file_path = f"{path}/{file}"
            image_features = self.encode_image(file_path)
            embeddings.append(image_features)
        synonyms = np.concatenate(embeddings)

        other_embeddings = []
        others = random.sample(other_files, len(synonym_files))
        for file in others:
            file_path = f"{path}/{file}"
            image_features = self.encode_image(file_path)
            other_embeddings.append(image_features)
        complement_embeddings = np.concatenate(other_embeddings)

        return synonyms, complement_embeddings, synonyms[0, :], complement_embeddings[0, :]
