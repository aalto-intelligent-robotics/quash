try:
    from embeddings_from_images.lseg.modules.models.lseg_net import LSegEncNet
    from embeddings_from_images.lseg.additional_utils.models import resize_image, pad_image, crop_image
except:
    import sys
    sys.path.append("embeddings_from_images")
    from lseg.modules.models.lseg_net import LSegEncNet
    from lseg.additional_utils.models import resize_image, pad_image, crop_image

import cv2

import os
import math
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

class PixelEmbeddingCreator:
    def __init__(self, vlmaps_model_path, crop_size):
        print("Loading model...")
        self.model = LSegEncNet(
            [""], arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size
        )
        model_state_dict = self.model.state_dict()
        pretrained_state_dict = torch.load(vlmaps_model_path)
        pretrained_state_dict = {
            k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()
        }
        model_state_dict.update(pretrained_state_dict)
        # model.load_state_dict(model_state_dict, strict=False)
        self.model.load_state_dict(pretrained_state_dict, strict=False)
        self.model.eval()
        # print("Model loaded.", cuda_info())
        self.model = self.model.cuda()

    def get_embeddings(self, rgb_image, crop_size=480, base_size=640):
        """
        Takes an RGB image as an input.

        Outputs a structure containing CLIP embeddings for every pixel in the image.
        """
        mask_version = 1  # 0, 1

        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        padding = [0.0] * 3
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        pix_feats = self.get_lseg_feat(rgb_image, [""], transform, crop_size, base_size, norm_mean, norm_std)

        pix_feats = pix_feats[0]
        pix_feats = pix_feats.transpose(1, 2, 0)

        return pix_feats

    def process_image(self, image_dir, image_file, embeddings_dir, parse_stamps, crop_size, base_size):
        image = self.read_image(f"{image_dir}/{image_file}")
        # print('Image read:', image.shape, image.dtype)

        # Get the embeddings
        image_embeddings = self.get_embeddings(image, crop_size, base_size)
        # print("Embeddings:", image_embeddings.shape)

        # Reshape the embeddings to [height x width x emb_size]

        print("Embeddings:", image_embeddings.shape, image_embeddings.dtype)

        # Save the embeddings
        if(parse_stamps):
            timestamp, height, width = image_file[: image_file.find(".")].split("_")
            embedding_file = (
                f"{embeddings_dir}/{timestamp}_{height}_{width}_{image_embeddings.shape[2]}.bin"
            )
        else:
            filename = image_file.replace("png", "bin")
            embedding_file = embeddings_dir + "/" + filename

        print("Saving embeddings to:", embedding_file)
        image_embeddings.tofile(embedding_file)
        print("Saved.")

    def read_image(self, image_path):
        return cv2.imread(image_path)

    def get_lseg_feat(self, image: np.array, labels, transform, crop_size=480, \
                 base_size=520, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):

        # shape before 1, 3, h, w
        # shape to h, w, 3
        image = transform(image).unsqueeze(0).cuda()
        # shape to h, w, 3
        img = image[0].permute(1,2,0)
        img = img * 0.5 + 0.5

        batch, _, h, w = image.size()
        stride_rate = 2.0/3.0
        stride = int(crop_size * stride_rate)

        long_size = base_size
        if h > w:
            height = long_size
            width = int(1.0 * w * long_size / h + 0.5)
            short_size = width
        else:
            width = long_size
            height = int(1.0 * h * long_size / w + 0.5)
            short_size = height

        cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})


        if long_size <= crop_size:
            pad_img = pad_image(cur_img, norm_mean,
                                norm_std, crop_size)
            with torch.no_grad():
                outputs, _ = self.model(pad_img, labels)
            outputs = crop_image(outputs, 0, height, 0, width)
        else:
            if short_size < crop_size:
                # pad if needed
                pad_img = pad_image(cur_img, norm_mean,
                                    norm_std, crop_size)
            else:
                pad_img = cur_img
            _,_,ph,pw = pad_img.shape #.size()
            assert(ph >= height and pw >= width)
            h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
            w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1

            with torch.cuda.device_of(image):
                with torch.no_grad():
                    outputs = image.new().resize_(batch, self.model.out_c,ph,pw).zero_().cuda()
                    # logits_outputs = image.new().resize_(batch, len(labels),ph,pw).zero_()
                count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
            # grid evaluation
            for idh in range(h_grids):
                for idw in range(w_grids):
                    h0 = idh * stride
                    w0 = idw * stride
                    h1 = min(h0 + crop_size, ph)
                    w1 = min(w0 + crop_size, pw)
                    crop_img = crop_image(pad_img, h0, h1, w0, w1)
                    # pad if needed
                    pad_crop_img = pad_image(crop_img, norm_mean,
                                                norm_std, crop_size)
                    with torch.no_grad():
                        output, _ = self.model(pad_crop_img, labels)
                    cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                    # cropped_logits = crop_image(logits, 0, h1-h0, 0, w1-w0)
                    outputs[:,:,h0:h1,w0:w1] += cropped
                    # logits_outputs[:,:,h0:h1,w0:w1] += cropped_logits
                    count_norm[:,:,h0:h1,w0:w1] += 1
            assert((count_norm==0).sum()==0)
            outputs = outputs / count_norm
            # logits_outputs = logits_outputs / count_norm
            outputs = outputs[:,:,:height,:width]
            # logits_outputs = logits_outputs[:,:,:height,:width]
        outputs = outputs.cpu()
        outputs = outputs.numpy() # B, D, H, W
        # predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
        # pred = predicts[0]

        return outputs