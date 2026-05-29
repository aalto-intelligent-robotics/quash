# VLMaps Embeddings for Images

This folder contains code for calculating CLIP embeddings for images in a given directory.

The code for embeddings is taken and adapted from the legacy VLMaps code from the `demo` branch of the [VLMaps project](https://github.com/vlmaps/vlmaps).

## Download model weights

VLMaps use LSeg model for image segmentation. 

To use it, please download the model weights file `demo_e200.ckpt`.

You can download them directly from [Google Drive](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view).

Or, alternatively, you can download it with the script:
```
pip install gdown
cd lseg
mkdir checkpoints
cd checkpoints
gdown 1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb
ls
```

## Download CLIP weights

Download the CLIP model "ViT-B/32".

## Run 

Execute the file `vlmaps_from_images.py` with the following arguments:

```
--img_dir - Directory to read images from.
--embeddings_dir - Directory to save embeddings to.
--clip_path - Path to saved CLIP weights (for "ViT-B/32").
--weights_path - Path to the LSeg model weights (the downloaded file demo_e200.ckpt).
```

```
python vlmaps_from_images.py --img_dir=/path/to/images --embeddings_dir=/path/to/embeddings
```
