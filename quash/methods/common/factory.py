import os
from pathlib import Path
from typing import Tuple, Optional
from methods.common.embedders import ImageEmbedder, TextEmbedder, EncoderType


def get_image_embedder(
    encoder: EncoderType,
    weights_path: str,
    temp_folder: str = "temp",
    cache: bool = True,
    device: str = "cuda",
    embedding_size: Optional[int] = None,
    crop_size: Optional[int] = None,
    base_size: Optional[int] = None,
    force_resize_images: bool = False,
    force_resize_size: Optional[Tuple[int, int]] = None,
    max_image_size: Optional[int] = None,
) -> ImageEmbedder:
    image_embedder = ImageEmbedder(
        encoder,
        weights_path,
        temp_folder,
        cache,
        device,
        embedding_size,
        crop_size,
        base_size,
        force_resize_images,
        force_resize_size,
        max_image_size,
    )
    return image_embedder


def get_text_embedder(
    device: str,
    network: str,
    model_name: str,
    tempfolder: str = "temp",
    cache: bool = True,
) -> TextEmbedder:
    text_embedder = TextEmbedder(device, network, model_name, tempfolder, cache)
    return text_embedder


def get_embedders(
    encoder: EncoderType,
    weights_path: str,
    device: str,
    network: str,
    model_name: str,
    tempfolder: str = "temp",
    cache: bool = True,
    embedding_size: Optional[int] = None,
    crop_size: Optional[int] = None,
    base_size: Optional[int] = None,
    force_resize_images: bool = False,
    force_resize_size: Optional[Tuple[int, int]] = None,
    max_image_size: Optional[int] = None,
) -> Tuple[ImageEmbedder, TextEmbedder]:
    image_embedder = get_image_embedder(
        encoder,
        weights_path,
        tempfolder,
        cache,
        device,
        embedding_size,
        crop_size,
        base_size,
        force_resize_images,
        force_resize_size,
        max_image_size,
    )
    text_embedder = get_text_embedder(device, network, model_name, tempfolder, cache)
    return image_embedder, text_embedder


def find(name, path) -> Optional[str]:
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return None


def get_lseg_weights_path() -> str:
    search_paths = [
        "/home/user/<path/to/quash>/weights",
        "/home/user/code/vlmaps/networks",
        "/home/user/<path/to/quash>/weights",
        "/home/user/code/vlmaps/networks",
    ]

    weights_path = None
    for path in search_paths:
        test_path = f"{path}/demo_e200.ckpt"
        if Path(test_path).exists():
            weights_path = test_path
            break

    if not weights_path:
        find_result = find("demo_e200.ckpt", "/home/user")
        if find_result:
            weights_path = find_result

    if not weights_path:
        weights_path = search_paths[0] + "/demo_e200.ckpt"

    return weights_path


def get_openseg_weights_path() -> str:
    search_paths = [
        "/home/user/<path/to/quash>/weights",
        "/home/user/code/vlmaps/networks",
        "/home/user/<path/to/quash>/weights",
        "/home/user/code/vlmaps/networks",
    ]

    weights_path = None
    for path in search_paths:
        test_path = f"{path}/openseg_exported_clip"
        if Path(test_path).exists():
            weights_path = test_path
            break

    if not weights_path:
        raise FileNotFoundError("Weight file not found")

    return weights_path


def get_default_embedders(
    encoder: EncoderType,
    tempfolder: str = "",
    cache: bool = True,
    force_resize_images: bool = False,
    force_resize_size: Optional[Tuple[int, int]] = None,
    max_image_size: Optional[int] = None,
):
    if not tempfolder:
        if Path("/home/user").exists():
            tempfolder = "/home/user/<path/to/quash>/temp"
        elif Path("/home/user").exists():
            tempfolder = "/home/user/<path/to/quash>/temp"
        else:
            tempfolder = "./temp"

    if encoder == EncoderType.LSEG:
        weights_path = get_lseg_weights_path()
        return get_embedders(
            encoder,
            weights_path,
            "cuda",
            "clip",
            "ViT-B/32",
            tempfolder,
            cache,
            None,
            480,  # crop size
            520,  # base size
            force_resize_images,
            force_resize_size,
            max_image_size,
        )
    elif encoder == EncoderType.OPENSEG:
        weights_path = get_openseg_weights_path()
        return get_embedders(
            encoder,
            weights_path,
            "cuda",
            "clip",
            "ViT-L/14@336px",
            tempfolder,
            cache,
            768,
            None,
            None,
            force_resize_images,
            force_resize_size,
            max_image_size,
        )
    else:
        raise ValueError("Unknown encoder type")
