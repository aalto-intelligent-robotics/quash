""" Returns an embedding from a VLM for a given text or image.
"""
import argparse
import numpy as np
try:
    from embeddings.embeddingCreator import EmbeddingCreator
except:
    from embeddingCreator import EmbeddingCreator

def main():
    """Main function.
    Prints in the console the embedding for the given text or image.
    If a file is specified, saves the embedding in binary format.
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text",
        dest="text",
        type=str,
        help="Text query."
    )
    group.add_argument(
        "--image_path",
        dest="image_path",
        type=str,
        help="Path to image used as a query."
    )
    parser.add_argument(
        "--embedding_file",
        dest="embedding_file",
        type=str,
        default=None,
        required=False,
        help="File in which to save the embedding in binary format. "
        + "If empty, the embedding is not saved to a file.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        required=False,
        help="Use cpu or cuda.",
    )
    parser.add_argument(
        "--network", "-n",
        dest="network",
        type=str,
        choices=["clip", "flava", "tbd"],
        required=True,
        help="Which NN model to use",
    )
    parser.add_argument(
        "--model_name", "-m",
        dest="model_name",
        type=str,
        default=None,
        required=True,
        help="Path to model",
    )
    args = parser.parse_args()
    device = args.device
    embedding_file = args.embedding_file
    text = args.text
    image_path = args.image_path
    network = args.network
    model_name = args.model_name

    creator = EmbeddingCreator(device, network, model_name)

    embedding: np.array
    if text:
        embedding = creator.get_text_embedding(text)
    elif image_path:
        embedding = creator.get_image_embedding(image_path)

    print(embedding)

    if embedding_file:
        embedding.tofile(embedding_file)
        print(f"Embedding saved to {embedding_file}")


if __name__ == "__main__":
    main()
