from abc import ABC, abstractmethod
from typing import Union, List
import clip
import torch


class Network(ABC):
    def __init__(self, device) -> None:
        self.device = device
        pass

    @abstractmethod
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def tokenize(
        self,
        texts: Union[str, List[str]],
        context_length: int = 77,
        truncate: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def preprocess(self, npx) -> torch.Tensor:
        pass


class Clip(Network):
    def __init__(self, device: str, model_name: str) -> None:
        super().__init__(device)
        self.model, self.preprocessor = clip.load(model_name, device=self.device)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(text)

    def tokenize(
        self,
        texts: Union[str, List[str]],
        context_length: int = 77,
        truncate: bool = False,
    ) -> torch.Tensor:
        return clip.tokenize(texts, context_length, truncate)

    def preprocess(self, npx) -> torch.Tensor:
        return self.preprocessor(npx)


class NetworkFactory:
    def __init__(self, device: str, network: str, model_name: str) -> None:
        self.network = network
        self.device = device
        self.model_name = model_name

    def createModel(self) -> Network:
        if self.network == "clip":
            return self.createClipModel(self.model_name)
        else:
            raise NotImplementedError("Only clip supported")

    def createClipModel(self, model_name: str) -> Network:
        model = Clip(self.device, model_name)
        return model
