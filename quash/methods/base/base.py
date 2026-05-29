import timeit
from typing import List, Tuple, Optional
from abc import abstractmethod
from methods.common.embedders import ImageEmbedder, TextEmbedder
import numpy as np
from methods.common.creators.visualEncoderFactory import (
    EncoderType,
)


class BasePredictor:
    def __init__(self, measure: bool = False) -> None:
        self.laps: List[float] = []
        self.measure = measure
        self.image_embedder: ImageEmbedder = ImageEmbedder(EncoderType.LSEG, "", "./temp")
        self.text_embedder: TextEmbedder = TextEmbedder("", "", "", "./temp")
        self.start = 0.0
        self.end = 0.0
        self.timer_title = ""
        self.image: np.ndarray = np.array(())

    def init_embedders(
        self,
        encoder: EncoderType,
        weights_path: str,
        device: str,
        network: str,
        model_name: str,
        tempfolder: str = "temp",
        cache: bool = True,
    ) -> None:
        # image processing
        self.image_embedder = ImageEmbedder(encoder, weights_path, tempfolder, cache)

        # query
        self.text_embedder = TextEmbedder(
            device, network, model_name, tempfolder, cache
        )

    def set_embedders(
        self, image_embedder: ImageEmbedder, text_embedder: TextEmbedder
    ) -> None:
        self.image_embedder = image_embedder
        self.text_embedder = text_embedder

    def start_timer(self, title: str = "") -> None:
        self.start = timeit.default_timer()
        self.laps.clear()
        self.timer_title = title

    def lap_timer(self) -> None:
        lap_end = timeit.default_timer()
        lap = lap_end - self.start
        self.laps.append(lap)

    def end_timer(self) -> None:
        self.end = timeit.default_timer()
        total = self.end - self.start
        print("Prediction time for", self.timer_title)
        if len(self.laps) > 0:
            i = 0
            for lap in self.laps:
                print("Lap", i, lap)
                i += 1
        print(f"Total: {total:.4f} s")

    def get_method_description(self) -> str:
        return ""

    @abstractmethod
    def predict_and_load_image(
        self,
        image_path: str,
        query: str,
        complement: List[str],
        environment: Optional[str] = None,
        synonyms: Optional[List[str]] = None,
        complement_synonyms: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[List[int]], np.ndarray, np.ndarray, np.ndarray]:
        pass
