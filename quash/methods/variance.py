from typing import Tuple, Optional, List, Dict
import numpy as np
from tqdm import tqdm
from PIL import Image
from methods.base.base import BasePredictor
from methods.common.metrics import cosine_distance_var
from methods.common.query_api import QueryAPI
from methods.common.logging import get_log_file
from methods.prompt import (
    synonym_prompt,
    antonym_prompt,
    synonym_system_prompt,
    antonym_system_prompt,
)

class VarianceWeighted(BasePredictor):
    def __init__(
        self,
        num_of_synonyms: int = 21,
        environment: str = "home",
        classifier: str = "svm",
        api_model: str = "gpt4-turbo",
        measure: bool = False,
        logpath: str = "",
        temp_folder: str = "../temp/",
        max_retries: int = int(1e3),
    ) -> None:
        super().__init__(measure)

        # parametrization
        self.environment = environment
        self.num_of_synonyms = num_of_synonyms
        self.classifier = classifier
        self.api_model = api_model
        self.dummy_mode = self.classifier == "dummy"

        self.query_cache: Dict[str, Tuple[List[str], List[str]]] = {}
        # logging
        self.logfile = get_log_file(logpath)
        # API
        self.queryAPI = QueryAPI(
            model=self.api_model, temp_folder=temp_folder, max_retries=max_retries
        )

    def predict_and_load_image(
        self,
        image_path: str,
        query: str,
        complement: List[str],
        environment: Optional[str] = None,
        synonyms: Optional[List[str]] = None,
        complement_synonyms: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[List[int]], np.ndarray, np.ndarray, np.ndarray]:
        if self.measure:
            self.start_timer("weighted")

        image_embeddings, image, width, height = self.image_embedder.read_and_embed(
            image_path
        )
        self.image = image

        mask, subtraction, distances, complement_distances = self._image_prediction(
            image_embeddings, query, complement, synonyms, complement_synonyms
        )

        mask = self.upscale(mask, width, height)

        if self.measure:
            self.end_timer()

        bboxes: List[List[int]] = [[]]

        return mask, bboxes, subtraction, distances, complement_distances

    def predict_image(
        self,
        image: np.ndarray,
        filename: str,
        query: str,
        complement: List[str],
        synonyms: list,
        complement_synonyms: list,
        base_size: int = 640,
        crop_size: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image_embeddings, _ = self.image_embedder.embed(
            image, filename, base_size, crop_size
        )
        return self._image_prediction(
            image_embeddings, query, complement, synonyms, complement_synonyms
        )

    def _image_prediction(
        self,
        image_embeddings: np.ndarray,
        query: str,
        complement: List[str],
        synonyms: Optional[List[str]] = None,
        complement_synonyms: Optional[List[str]] = None,
    ):
        height = image_embeddings.shape[0]
        width = image_embeddings.shape[1]
        depth = image_embeddings.shape[2]
        image_embeddings = image_embeddings.reshape(-1, depth)

        match, subtraction, distances, complement_distances = self.predict(
            image_embeddings, query, complement, synonyms, complement_synonyms
        )

        match = match.reshape((height, width))
        subtraction = subtraction.reshape((height, width))
        distances = distances.reshape((height, width))
        complement_distances = complement_distances.reshape((height, width))
        return match, subtraction, distances, complement_distances

    def predict(
        self,
        embeddings: np.ndarray,
        query: str,
        complement: List[str],
        synonyms: Optional[List[str]] = None,
        complement_synonyms: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        query_embedding = self.text_embedder.embed(query)
        complement_embeddings_l = []
        for comp in complement:
            complement_embedding = self.text_embedder.embed(comp)
            complement_embeddings_l.append(complement_embedding)
        complement_embeddings = np.array(complement_embeddings_l)

        if not synonyms:
            synonyms, complement_synonyms = self.get_synonyms(query)

        assert synonyms
        assert complement_synonyms

        synonyms.append(query)
        synonym_embeddings_l = []
        for s in tqdm(synonyms, leave=False, desc="Synonyms"):
            synonym_embeddings_l.append(self.text_embedder.embed(s))
        synonym_embeddings = np.array(synonym_embeddings_l)
        variance = synonym_embeddings.var(axis=0)

        for comp in complement:
            if comp not in complement_synonyms:
                complement_synonyms.append(comp)
        complement_synonym_embeddings_l = []
        for s in tqdm(complement_synonyms, leave=False, desc="Antonyms"):
            complement_synonym_embeddings_l.append(self.text_embedder.embed(s))
        complement_synonym_embeddings = np.array(complement_synonym_embeddings_l)
        complement_variance = complement_synonym_embeddings.var(axis=0)

        length = embeddings.shape[0]
        distances = np.zeros((length))
        complement_distances = np.zeros((length, len(complement)))

        for i in range(length):
            embedding = embeddings[i, :]
            query_distance = cosine_distance_var(embedding, query_embedding, variance)
            distances[i] = query_distance
            for j, complement_embedding in enumerate(complement_embeddings):
                complement_distance = cosine_distance_var(
                    embedding, complement_embedding, complement_variance
                )
                complement_distances[i, j] = complement_distance

        # subtraction
        complement_distances = np.array(complement_distances)
        complement_distances = np.max(complement_distances, axis=1)
        subtraction = distances - complement_distances

        # prediction
        match = distances >= complement_distances
        match = match.astype(np.int32)

        return match, subtraction, distances, complement_distances

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

    def get_synonyms(self, query: str) -> Tuple[List[str], List[str]]:
        if query in self.query_cache:
            return self.query_cache[query]

        s_prompt, a_prompt = self.get_prompts(query)
        synonyms_raw = self.queryAPI.query(synonym_system_prompt, s_prompt)
        synonyms = synonyms_raw.split(",")
        for i, synonym in enumerate(synonyms):
            synonyms[i] = synonym.strip()

        antonyms_raw = self.queryAPI.query(antonym_system_prompt, a_prompt)
        antonyms = antonyms_raw.split(",")
        for i, antonym in enumerate(antonyms):
            antonyms[i] = antonym.strip()

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

    def get_prompts(self, query: str) -> Tuple[str, str]:
        s_prompt = (
            synonym_prompt.replace("<query>", query)
            .replace("<env>", self.environment)
            .replace("<num>", str(self.num_of_synonyms))
        )
        a_prompt = (
            antonym_prompt.replace("<query>", query)
            .replace("<env>", self.environment)
            .replace("<num>", str(self.num_of_synonyms))
        )
        return s_prompt, a_prompt
