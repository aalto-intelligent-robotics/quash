import torch
import numpy as np
from visualEncoderFactory import VisualEncoderFactory, VisualEncoder


class PixelEmbeddingCreator:
    def __init__(
        self,
        visual_encoder_name: str,
        device: str,
        saved_model_path: str,
        embedding_size: int,
        crop_size: int,
        base_size: int,
    ):
        self.device = device
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        factory = VisualEncoderFactory()
        self.model: VisualEncoder = factory.createVisualEncoder(
            visual_encoder_name, saved_model_path, embedding_size, crop_size, base_size
        )

    def get_image_features(
        self,
        image_path: str,
    ) -> np.ndarray:
        """Return the embedding for a given image file."""
        return self.model.get_image_features(image_path)
