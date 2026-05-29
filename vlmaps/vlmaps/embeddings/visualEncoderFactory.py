from abc import ABC
import numpy as np
import torch
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import os
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
import gdown
import cv2

from vlmaps.utils.lseg_utils import get_lseg_feat
from vlmaps.lseg.modules.models.lseg_net import LSegEncNet


class VisualEncoder(ABC):
    def __init__(self, device: str):
        pass

    def get_image_features(self, image_path: str) -> np.ndarray:
        pass


class OpenSegEncoder(VisualEncoder):
    def __init__(self, device: str, saved_model_path: str, embedding_size: int):
        self.device = device
        # Initialize the model from the saved weights
        self.model = tf2.saved_model.load(
            saved_model_path,
            tags=[tf.saved_model.tag_constants.SERVING],
        )
        # Building a text embedding for OpenSeg
        # The text embedding is not needed for extracting image features,
        # therefore, using a zero embedding
        self.zero_text_emb = tf.zeros([1, 1, embedding_size])

    def get_image_features(
        self, image_path: str, image_size=None, regional_pool=True
    ) -> np.ndarray:
        # Read image from file
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # run OpenSeg
        results = self.model.signatures["serving_default"](
            inp_image_bytes=tf.convert_to_tensor(image_bytes),
            inp_text_emb=self.zero_text_emb,
        )
        img_info = results["image_info"]
        crop_sz = [
            int(img_info[0, 0] * img_info[2, 0]),
            int(img_info[0, 1] * img_info[2, 1]),
        ]
        if regional_pool:
            image_embedding_feat = results["ppixel_ave_feat"][
                :, : crop_sz[0], : crop_sz[1]
            ]
        else:
            image_embedding_feat = results["image_embedding_feat"][
                :, : crop_sz[0], : crop_sz[1]
            ]
        if image_size is not None:
            feat_2d = tf.cast(
                tf.image.resize_nearest_neighbor(
                    image_embedding_feat, image_size, align_corners=True
                )[0],
                dtype=tf.float16,
            ).numpy()
        else:
            feat_2d = tf.cast(
                image_embedding_feat[[0]], dtype=tf.float16
            ).numpy()

        # feat_2d = np.expand_dims(feat_2d, 0)
        feat_2d = (
            torch.from_numpy(feat_2d)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .cpu()
            .numpy()
        )

        return feat_2d


class LSegEncoder(VisualEncoder):
    def __init__(
        self, device: str, saved_model_path: str, crop_size: int, base_size: int
    ):
        # Code from vlmap_builder.init_lseg()
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        lseg_model = LSegEncNet(
            "",
            arch_option=0,
            block_depth=0,
            activation="lrelu",
            crop_size=crop_size,
        )
        model_state_dict = lseg_model.state_dict()
        # checkpoint_dir = (
        #     Path(__file__).resolve().parents[1] / "lseg" / "checkpoints"
        # )
        # checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
        # print("checkpoint", checkpoint_path)
        # os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = saved_model_path
        if not os.path.exists(checkpoint_path):
            print("Downloading LSeg checkpoint...")
            # the checkpoint is from official LSeg github repo
            # https://github.com/isl-org/lang-seg
            checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
            gdown.download(checkpoint_url, output=str(checkpoint_path))

        pretrained_state_dict = torch.load(
            checkpoint_path, map_location=self.device
        )
        pretrained_state_dict = {
            k.lstrip("net."): v
            for k, v in pretrained_state_dict["state_dict"].items()
        }
        model_state_dict.update(pretrained_state_dict)
        lseg_model.load_state_dict(pretrained_state_dict)

        lseg_model.eval()
        lseg_model = lseg_model.to(self.device)
        self.lseg_model = lseg_model

        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]
        self.lseg_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.embedding_size = lseg_model.out_c
        self.crop_size = crop_size
        self.base_size = base_size

    def get_image_features(self, image_path: str) -> np.ndarray:
        bgr = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pix_feats = get_lseg_feat(
            self.lseg_model,
            rgb,
            ["example"],
            self.lseg_transform,
            self.device,
            self.crop_size,
            self.base_size,
            self.norm_mean,
            self.norm_std,
        )

        return pix_feats


class VisualEncoderFactory:
    def __init__(self, device: str, visual_encoder: str):
        self.device = device
        self.visual_encoder = visual_encoder

    def createVisualEncoder(
        self,
        saved_model_path: str,
        embedding_size: int,
        crop_size: int,
        base_size: int,
    ) -> VisualEncoder:
        self.saved_model_path = saved_model_path
        if self.visual_encoder == "LSeg":
            return self.createLSegEncoder(crop_size, base_size)
        if self.visual_encoder == "OpenSeg":
            return self.createOpenSegEncoder(embedding_size)

    def createLSegEncoder(self, crop_size, base_size):
        model = LSegEncoder(
            self.device, self.saved_model_path, crop_size, base_size
        )
        return model

    def createOpenSegEncoder(self, embedding_size: int):
        model = OpenSegEncoder(
            self.device, self.saved_model_path, embedding_size
        )
        return model
