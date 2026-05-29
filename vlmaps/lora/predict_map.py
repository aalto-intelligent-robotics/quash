from argparse import ArgumentParser
from typing import List, Union
from transformer_model import (
    TransformerModel,
    VectorTransformer,
    LoraAggregator,
)
from training_dataset import Dataset, get_batches
from vlmaps_dataset import VLMapsDataset
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


def main(maps: List[str] | List[int], encoder: str, new_model: bool, dim: int, max_tokens: int, save_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if new_model:
        model = VectorTransformer.load(save_path, dim).to(device)
        maptype = "new"
    else:
        model = LoraAggregator.load(save_path, dim, max_tokens).to(device)
        maptype = "lora"
    with open(f"{save_path}/metadata.out", "r") as f:
        lines = f.readlines()
        metadata = "".join(lines)

    data_path = "/home/user/hdd/datasets/vlmaps_dataset"
    class_path = "/home/user/code/vlmaps/cfg/mpcat40.tsv"
    dataset = VLMapsDataset(
        data_path,
        class_path,
        encoder,
        maps,
        None,
        None,
        "cuda",
        write_mode=True
    )
    model.eval()

    with torch.no_grad():
        map_data = dataset.get_maps()
        for single_map_data in tqdm(map_data, desc="Maps"):
            map, embeddings, histogram = single_map_data
            new_embeddings = np.copy(embeddings)
            for key, value in tqdm(histogram.items(), desc="Histogram", leave=False):
                if key == -1:
                    continue
                T = torch.Tensor(np.array(value)).unsqueeze(0).to(device)

                if new_model:
                    pred = model(T)
                else:
                    if T.shape[1] < dim:
                        pred = model(T)
                    else:
                        pred = LoraAggregator.bert_forward_with_embeds(model, T)  # type: ignore
                new_embeddings[key] = pred.detach().cpu().numpy()

            np.save(f"{map}/predicted_embeddings_{maptype}.npy", new_embeddings)
            with open(f"{map}/prediction_metadata_{maptype}.out", "w") as f:
                lines = []
                lines.append(f"maps: {maps}\n")
                lines.append(f"data_path: {data_path}\n")
                lines.append(f"class_path: {class_path}\n")
                lines.append(f"new_model: {new_model}\n")
                lines.append(f"dim: {dim}\n")
                lines.append(f"save_path: {save_path}\n")
                lines.append("\n")
                lines.append("Training metadata:\n")
                lines.append(metadata)
                f.writelines(lines)
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--new", action="store_true", help="Train a new model from scratch"
    )
    parser.add_argument("--dim", type=int, help="Embedding dimension", default=512)
    parser.add_argument("--max_tokens", type=int, help="Max tokens", default=512)
    parser.add_argument("--path", type=str, help="Save/load path")
    parser.add_argument("--maps", nargs="+", help="User maps", default=[])
    parser.add_argument("--encoder", type=str, help="Encoder", required=True)

    args = parser.parse_args()
    if not args.path:
        if args.new:
            save_path = "vector_transformer"
        else:
            save_path = "aggregator_lora"
    else:
        save_path = args.path

    parsed_maps: List[int] | List[str] = []
    if args.maps:
        try:
            for item in args.maps:
                parsed_maps.append(int(item))  # type: ignore
        except ValueError:
            parsed_maps = args.maps

    main(
        maps=parsed_maps,
        encoder=args.encoder,
        new_model=args.new,
        dim=args.dim,
        max_tokens=args.max_tokens,
        save_path=args.path,
    )
