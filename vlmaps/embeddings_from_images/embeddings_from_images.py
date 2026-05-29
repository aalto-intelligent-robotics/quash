import argparse
import os
try:
    from embeddings_from_images.pixelEmbeddingCreator import PixelEmbeddingCreator
except:
    from pixelEmbeddingCreator import PixelEmbeddingCreator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        dest="img_dir",
        type=str,
        help="Directory to read images from.",
        required=True
    )
    parser.add_argument(
        "--embeddings_dir",
        dest="embeddings_dir",
        type=str,
        help="Directory to save embeddings to.",
        required=True,
    )
    parser.add_argument(
        "--weights_path",
        dest="weights_path",
        type=str,
        help="Path to the LSeg model weights.",
        required=True,
    )
    parser.add_argument(
        "--crop_size",
        dest="crop_size",
        type=int,
        help="crop_size of the images",
        required=True
    )
    parser.add_argument(
        "--base_size",
        dest="base_size",
        type=int,
        help="base_size of the images",
        required=True
    )
    parser.add_argument(
        "--parse_stamps",
        dest="parse_stamps",
        action="store_true",
        help="Parse timestamp from image names",
        required=False,
        default=False
    )
    args = parser.parse_args()

    img_dir = args.img_dir
    embeddings_dir = args.embeddings_dir
    weights_path = args.weights_path
    parse_stamps = args.parse_stamps
    crop_size  = args.crop_size
    base_size = args.base_size

    creator = PixelEmbeddingCreator(weights_path, crop_size)

    print("Reading images from", img_dir)
    for img_file in sorted(os.listdir(img_dir)):
        print("Processing image", img_file)
        creator.process_image(img_dir, img_file, embeddings_dir, parse_stamps, crop_size, base_size)
