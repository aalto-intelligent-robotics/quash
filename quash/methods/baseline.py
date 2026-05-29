from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
from methods.base.base import BasePredictor
from methods.common.metrics import MetricType, matrix_distance, matrix_l2


class Baseline(BasePredictor):
    def __init__(
        self, measure: bool = False, metric_type: MetricType = MetricType.COSINE
    ) -> None:
        super().__init__(measure)
        self.image = np.array([])
        self.metric_type = metric_type
        self.metric_function = None

    def set_metric_function(self, fx) -> None:
        self.metric_function = fx

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
            self.start_timer("baseline")

        image_embeddings, image, width, height = self.image_embedder.read_and_embed(
            image_path
        )
        self.image = image

        mask, subtraction, distances, complement_distances = self._image_prediction(
            image_embeddings, query, complement
        )

        mask = self.upscale(mask, width, height)

        # TODO
        bboxes: List[List[int]] = [[]]

        if self.measure:
            self.end_timer()

        return mask, bboxes, subtraction, distances, complement_distances

    def predict_image(
        self,
        image: np.ndarray,
        filename: str,
        query: str,
        complement: List[str],
        base_size: int = 640,
        crop_size: int = -1,
    ):
        image_embeddings, image = self.image_embedder.embed(
            image, filename, base_size, crop_size
        )
        self.image = image

        match, subtraction, distances, complement_distances = self._image_prediction(
            image_embeddings, query, complement
        )

        return match, subtraction, distances, complement_distances

    def _image_prediction(
        self,
        image_embeddings: np.ndarray,
        query: str,
        complement: List[str],
    ):
        height = image_embeddings.shape[0]
        width = image_embeddings.shape[1]
        depth = image_embeddings.shape[2]
        image_embeddings = image_embeddings.reshape(-1, depth)

        match, subtraction, distances, complement_distances = self.predict(
            image_embeddings, query, complement
        )

        match = match.reshape((height, width))
        subtraction = subtraction.reshape((height, width))
        distances = distances.reshape((height, width))
        complement_distances = complement_distances.reshape((height, width))

        return match, subtraction, distances, complement_distances

    def predict(
        self, embeddings: np.ndarray, query: str, complement: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        query_embedding = self.text_embedder.embed(query)
        complement_embeddings_l = []
        for comp in complement:
            complement_embedding = self.text_embedder.embed(comp)
            complement_embeddings_l.append(complement_embedding)
        complement_embeddings = np.array(complement_embeddings_l)

        length = embeddings.shape[0]
        distances = np.zeros((length))
        complement_distances = np.zeros((length))

        qe = query_embedding.reshape((1, -1))
        ce = complement_embeddings
        texts = np.concatenate((qe, ce), axis=0)
        if self.metric_type == MetricType.COSINE:
            all_distances = matrix_distance(embeddings, texts)
        elif self.metric_type == MetricType.EUCLIDEAN:
            all_distances = matrix_l2(embeddings, texts)
        else:
            assert self.metric_function is not None, "metric function not set"
            all_distances = self.metric_function(embeddings, texts)
        distances = all_distances[:, 0]
        complement_distances = all_distances[:, 1:].T
        # for i in tqdm(range(length), leave=False, desc="Baseline predict"):
        #     embedding = embeddings[i, :]
        #     query_distance = cosine_distance(embedding, query_embedding)
        #     distances[i] = query_distance
        #     complement_distance = cosine_distance(embedding, complement_embedding)
        #     complement_distances[i] = complement_distance

        if len(complement) > 1:
            complement_distances = np.max(complement_distances, axis=0)

            # subtraction
            subtraction = distances - complement_distances

            # prediction
            indices = np.argmax(all_distances, axis=1)
            match = indices == 0
            match = match.astype(np.int32)

            return match, subtraction, distances, complement_distances
        else:
            # subtraction
            subtraction = distances - complement_distances

            # prediction
            if self.metric_type == MetricType.COSINE:
                match = distances >= complement_distances
            else:
                match = distances <= complement_distances
            match = match.astype(np.int32)

            return match, subtraction, distances, complement_distances

    def get_method_description(self) -> str:
        return "Our LSeg"

    def upscale(self, mask: np.ndarray, width: int, height: int) -> np.ndarray:
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
