import os
import uuid
import math
from typing import Tuple, Optional
import numpy as np
import cv2
from methods.common.creators.visualEncoderFactory import (
    VisualEncoderFactory,
    VisualEncoder,
    EncoderType
)
from methods.common.creators.embeddingCreator import EmbeddingCreator
from methods.common.cache_manager import CacheManager


class ImageEmbedder:
    def __init__(
        self,
        encoder: EncoderType,
        weights_path: str,
        temp_folder: str,
        cache: bool = True,
        device: str = "cuda",
        embedding_size: Optional[int] = None,
        crop_size: Optional[int] = None,
        base_size: Optional[int] = None,
        force_resize_images: bool = False,
        force_resize_size: Optional[Tuple[int, int]] = None,
        max_image_size: Optional[int] = None,
    ) -> None:
        self.cache_manager = CacheManager(f"{temp_folder}/img", max_size=100000)
        self.weights_path = weights_path
        self.creator_created = False
        self.cache = cache
        self.device = device
        self.id = str(uuid.uuid4())
        self.encoder = encoder
        self.embedding_size = embedding_size
        self.crop_size = crop_size
        self.base_size = base_size
        self.force_resize_images = force_resize_images
        self.force_resize_size = force_resize_size
        if max_image_size is None or max_image_size < 0:
            self.max_image_size = None
        else:
            self.max_image_size = max_image_size

        self.model: Optional[VisualEncoder] = None

    def get_embedder(self) -> VisualEncoder:
        if not self.model:
            self.model = VisualEncoderFactory.createVisualEncoder(
                self.encoder,
                self.device,
                self.weights_path,
                self.embedding_size,
                self.crop_size,
                self.base_size,
            )
        return self.model

    def read(self, image_path: str) -> Tuple[np.ndarray, int, int, str, int]:
        image: np.ndarray = cv2.imread(image_path)  # type: ignore
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        image = np.stack((b, g, r), axis=2)
        width = image.shape[1]
        height = image.shape[0]
        self.image = image
        filename = os.path.basename(image_path)
        if self.force_resize_images:
            image = cv2.resize(image, self.force_resize_size)  # type: ignore
        else:
            image = self.resize_image(image)

        dim = max(image.shape[0], image.shape[1])
        return image, width, height, filename, dim

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        # Check if resizing is necessary
        if self.max_image_size is None:
            return image

        # Get the original dimensions of the image
        height, width = image.shape[:2]

        # Resize only if either dimension exceeds the self.max_image_size
        if max(height, width) <= self.max_image_size:
            return image

        # Determine the scaling factor
        if width > height:  # If width is larger
            scale = self.max_image_size / float(width)
        else:  # If height is larger or equal
            scale = self.max_image_size / float(height)

        new_width = int(width * scale)
        new_height = int(height * scale)
        # Resize the image while maintaining aspect ratio
        resized_image = cv2.resize(image, (new_width, new_height))  # type: ignore

        return resized_image

    def read_and_embed(
        self, image_path: str
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        image, width, height, _, _ = self.read(image_path)
        embeddings = self.embed(image, image_path)
        return embeddings, image, width, height

    def embed(
        self,
        image: np.ndarray,
        filename: str,
        base_size: int = -1,
        crop_size: int = -1,
    ) -> np.ndarray:
        if base_size == -1:
            base_size = image.shape[1]

        if crop_size == -1:
            crop_size = math.floor(base_size / 32) * 32

        if crop_size == base_size:
            crop_size = math.floor((base_size * 0.75) / 32) * 32

        if not self.base_size:
            self.base_size = base_size
        if not self.crop_size:
            self.crop_size = crop_size

        filepath = f"{filename.replace('/', '_')}-{self.encoder}.npy"
        if self.cache:
            embeddings = self.cache_manager.load(filepath)
            if embeddings is not None:
                return embeddings

        embedder = self.get_embedder()
        embeddings = embedder.read_and_embed(filename)
        if self.cache:
            self.cache_manager.save(filepath, embeddings)
        return embeddings


class TextEmbedder:
    def __init__(
        self,
        device: str,
        network: str,
        model_name: str,
        temp_folder: str,
        cache: bool = True,
    ) -> None:
        self.creator_created = False
        self.device = device
        self.network = network
        self.model_name = model_name
        self.cache_manager = CacheManager(f"{temp_folder}/text", max_size=100000)
        self.cache = cache

    def embed(self, text: str) -> np.ndarray:
        filepath = (
            f"{text.replace('/', '_')}"
            f"_{self.network}_{self.model_name.replace('/', '_')}.npy"
        )

        if self.cache:
            embeddings = self.cache_manager.load(filepath)
            if embeddings is not None:
                return embeddings

        if not self.creator_created:
            self.creator = EmbeddingCreator(
                self.device, self.network, self.model_name
            )
            self.creator_created = True
        embeddings = self.creator.get_text_embedding(text)
        if self.cache:
            self.cache_manager.save(filepath, embeddings)
        return embeddings
