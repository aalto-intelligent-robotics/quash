import torch
import cv2
from PIL import Image
import numpy as np

# this is for vscode debugging
try:
    from networkFactory import NetworkFactory
except:
    from embeddings.networkFactory import NetworkFactory

class EmbeddingCreator:
    def __init__(self, device, network = "clip", model_name = "ViT-B/32"):
        self.device = device
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        factory = NetworkFactory(self.device, network, model_name)
        self.model = factory.createModel()

    def get_image_embedding(self,
            image_path: str,
        ) -> np.array:
        """Return the embedding for a given image file.

        Args:
            image_path: Path to the image query
            model: CLIP model
            preprocess: CLIP image preprocessor
            device: Torch device (cpu or cuda)

        Returns:
            The embedding of the given image as a numpy float32 array.
        """
        image = cv2.imread(image_path)
        print("image", image.shape)
        image_ready = self.model.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding: torch.FloatTensor = self.model.encode_image(
                image_ready).to(self.device)

        return embedding.squeeze().cpu().numpy().astype("float32")


    def get_text_embedding(self, text: str) -> np.array:
        """Return the embedding for a given text.

        Args:
            text: Text query
            model: CLIP model
            device: Torch device (cpu or cuda)

        Returns:
            The embedding of the given text as a numpy float32 array.
        """
        text_tokens = self.model.tokenize([text]).to(self.device)
        with torch.no_grad():
            embedding: torch.FloatTensor = self.model.encode_text(
                text_tokens).to(self.device)

        return embedding.squeeze().cpu().numpy().astype("float32")
