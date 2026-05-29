import torch
import numpy as np

# this is for vscode debugging
try:
    from visualEncoderFactory import VisualEncoderFactory, VisualEncoder
except:
    from embeddings.visualEncoderFactory import (
        VisualEncoderFactory,
        VisualEncoder,
    )


class PixelEmbeddingCreator:
    def __init__(
        self,
        device: str,
        visual_encoder_name: str,
        saved_model_path: str,
        embedding_size: int,
        crop_size: int,
        base_size: int,
    ):
        self.device = device
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        factory = VisualEncoderFactory(self.device, visual_encoder_name)
        self.model: VisualEncoder = factory.createVisualEncoder(
            saved_model_path, embedding_size, crop_size, base_size
        )

    def get_image_features(
        self,
        image_path: str,
    ) -> np.array:
        """Return the embedding for a given image file."""
        return self.model.get_image_features(image_path)
