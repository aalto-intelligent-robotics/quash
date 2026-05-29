import clip
from transformers import FlavaImageProcessor, FlavaModel, BertTokenizer
from abc import ABC


class Network(ABC):
    def __init__(self, device):
        self.device = device
        pass

    def encode_image(self, image):
        pass

    def encode_text(self, text):
        pass

    def tokenize(self, texts):
        pass


class Clip(Network):
    def __init__(self, device, model_name):
        super().__init__(device)
        self.model, self.preprocessor = clip.load(model_name, device=self.device)

    def encode_image(self, image):
        return self.model.encode_image(image)

    def encode_text(self, text):
        return self.model.encode_text(text)

    def tokenize(self, texts, context_length=77, truncate=False):
        return clip.tokenize(texts, context_length, truncate)

    def preprocess(self, npx):
        return self.preprocessor(npx)


class Flava(Network):
    def __init__(self, device, model_name):
        super().__init__(device)
        self.model = FlavaModel.from_pretrained(model_name).to(device)
        self.image_processor = FlavaImageProcessor.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def encode_image(self, image):
        # The first token is the [CLS_I], which should contain
        # the representation of the full image.
        return self.model.get_image_features(image)[0][0]

    def encode_text(self, text):
        # The first token is the [CLS], which should contain
        # the representation of the full text.
        return self.model.get_text_features(**text)[0][0]

    def tokenize(self, texts, return_tensors="pt"):
        return self.tokenizer(text=texts, return_tensors=return_tensors)

    def preprocess(self, npx, return_tensors="pt"):
        preprocessed = self.image_processor.preprocess(
            images=[npx], return_tensors=return_tensors
        )
        return preprocessed["pixel_values"].squeeze(0)


class NetworkFactory:
    def __init__(self, device, network, model_name):
        self.network = network
        self.device = device
        self.model_name = model_name

    def createModel(self):
        if self.network == "clip":
            return self.createClipModel(self.model_name)
        if self.network == "flava":
            return self.createFlavaModel(self.model_name)

    def createClipModel(self, model_name):
        model = Clip(self.device, model_name)
        return model

    def createFlavaModel(self, model_name):
        model = Flava(self.device, model_name)
        return model
