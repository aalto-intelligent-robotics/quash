import torch
from typing import List, Union, Optional
import torch.nn.functional as F
import open_clip
from transformer_model import (
    TransformerModel,
    VectorTransformer,
    LoraAggregator,
)
from training_dataset import Dataset, get_batches, get_batches_by_label, get_group_by_label
from vlmaps_dataset import VLMapsDataset
from tqdm import tqdm
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import argparse
import wandb
import random
import signal
import sys

G_RUNNING = True

def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    global G_RUNNING
    G_RUNNING = False


# ███████╗██╗   ██╗ █████╗ ██╗
# ██╔════╝██║   ██║██╔══██╗██║
# █████╗  ██║   ██║███████║██║
# ██╔══╝  ╚██╗ ██╔╝██╔══██║██║
# ███████╗ ╚████╔╝ ██║  ██║███████╗
# ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝


@torch.no_grad()
def evaluate_toy_aggregator(
    device: str,
    word_group: List[str],
    candidate_labels: List[str],
    model: TransformerModel,
    clip_model,
    tokenizer,
    topk=5,
):
    model.eval()
    tokens = tokenizer(word_group).to(device)
    word_embeddings = clip_model.encode_text(tokens)
    word_embeddings = F.normalize(word_embeddings, dim=-1).unsqueeze(0)  # (1, T, D)

    # --- Aggregated via transformer ---
    aggregated = model(word_embeddings)  # (1, D)
    aggregated = F.normalize(aggregated, dim=-1)

    # --- Candidate embeddings ---
    cand_tokens = tokenizer(candidate_labels).to(device)
    cand_embs = clip_model.encode_text(cand_tokens)
    cand_embs = F.normalize(cand_embs, dim=-1)  # (C, D)

    # --- Aggregated similarity ---
    sims = torch.matmul(aggregated, cand_embs.T).squeeze()  # (C,)
    top_indices = torch.topk(sims, k=topk).indices.tolist()

    print(f"\n🔍 Input words: {word_group}")
    print(f"📌 Top-{topk} decoded labels (aggregated):")
    for i in top_indices:
        label = candidate_labels[i]
        score = sims[i].item()
        print(f"{label:>10s}  —  similarity: {score:.3f}")

    # --- Baseline: mean of raw CLIP embeddings ---
    baseline = F.normalize(
        word_embeddings[0].mean(dim=0, keepdim=True), dim=-1
    )  # (1, D)
    baseline_sims = torch.matmul(baseline, cand_embs.T).squeeze()
    baseline_top = torch.topk(baseline_sims, k=topk).indices.tolist()

    print("\n📎 Baseline (mean of CLIP embeddings):")
    for i in baseline_top:
        label = candidate_labels[i]
        score = baseline_sims[i].item()
        print(f"{label:>10s}  —  similarity: {score:.3f}")


@torch.no_grad()
def eval(
    model: TransformerModel, device: str, dataset: VLMapsDataset, plot: bool = True
):
    model.eval()
    # Test query
    while True:
        query = input("\nEnter query word, or 'q' to quit: ").strip()
        if query.lower() == "q":
            break
        idx = dataset.word2idx(query)
        embeddings = dataset.get_embeddings(idx)

        if embeddings.shape[0] == 0:
            print("no embeddings found")
            continue

        query_embedding = dataset.get_class_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)

        T = torch.Tensor(embeddings).unsqueeze(0).to(device)
        predictions = model(T).detach().cpu().numpy()
        sims = cosine_similarity(predictions, query_embedding).flatten()
        bl_sims = cosine_similarity(embeddings, query_embedding).flatten()
        stats(sims, bl_sims, plot)


def stats(sims: np.ndarray, bl_sims: np.ndarray, plot: bool):
    mean, max, min = sims.mean(), sims.max(), sims.min()
    print(f"Predicted: {min:>9.5f} {mean:>9.5f} {max:>9.5f}")

    bl_mean, bl_max, bl_min = bl_sims.mean(), bl_sims.max(), bl_sims.min()
    print(f"Baseline : {bl_min:>9.5f} {bl_mean:>9.5f} {bl_max:>9.5f}")

    print(f"Diff     : {min-bl_min:>9.5f} {mean-bl_mean:>9.5f} {max-bl_max:>9.5f}")

    if plot:
        _, ax = plt.subplots(1, 2, figsize=(15, 10))
        ax[0].hist(sims, bins=30)
        ax[0].set_title("Predicted")
        ax[0].set_xlabel("Cosine similarity")
        ax[0].set_ylabel("Frequency")
        ax[0].grid(True)

        ax[1].hist(bl_sims, bins=30)
        ax[1].set_title("Baseline")
        ax[1].set_xlabel("Cosine similarity")
        ax[1].set_ylabel("Frequency")
        ax[1].grid(True)

        plt.show()


# ████████╗██████╗  █████╗ ██╗███╗   ██╗
# ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║
#    ██║   ██████╔╝███████║██║██╔██╗ ██║
#    ██║   ██╔══██╗██╔══██║██║██║╚██╗██║
#    ██║   ██║  ██║██║  ██║██║██║ ╚████║
#    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝


def train(
    epochs: int,
    new_model: bool,
    dim: int,
    N: int,
    max_tokens: int,
    batch_size: int,
    device: str,
    training_dataset: Dataset,
    validation_dataset: Dataset,
    intermediate_save: int,
    save_path: str,
    debug: bool,
    name: Optional[str],
):
    model: TransformerModel
    if new_model:
        model = VectorTransformer(dim=dim).to(device)
    else:
        model = LoraAggregator(dim=dim, max_position_embeddings=max_tokens).to(device)
    # debug
    print("\n🔍 Trainable Parameters:")
    for named_p, p in model.named_parameters():
        if p.requires_grad:
            print("✅", named_p, p.shape)

    for named_p, param in model.named_parameters():
        if not any(x in named_p for x in ["lora", "cls_token", "proj"]):
            param.requires_grad = False

    # optimizer
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )

    wandb.login()
    project = "transformer" if not debug else "debug"
    config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "new_model": new_model,
        "dim": dim,
        "N": N,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "device": device,
        "dataset": training_dataset.get_description(),
        "intermediate_save": intermediate_save,
        "save_path": save_path,
    }

    global G_RUNNING
    # training loop
    with wandb.init(name=name, project=project, config=config) as run:
        for epoch in tqdm(range(epochs), desc="Epochs"):
            if not G_RUNNING:
                break

            loss = train_one_epoch(
                model, optimizer, training_dataset, batch_size, N, device, validate=False
            )

            if not G_RUNNING:
                break

            validation_loss = train_one_epoch(
                model, optimizer, validation_dataset, batch_size, N, device, validate=True
            )
            run.log({"validation_loss": validation_loss, "loss": loss})
            if intermediate_save > 0 and epoch > 0 and epoch % intermediate_save == 0:
                model.save(f"{save_path}/intermediate", True)
                save_metadata(
                    model,
                    epochs,
                    new_model,
                    dim,
                    N,
                    max_tokens,
                    batch_size,
                    device,
                    training_dataset,
                    f"{save_path}/intermediate",
                )

    return model


def save_metadata(
    model: TransformerModel,
    epochs: int,
    new_model: bool,
    dim: int,
    N: int,
    max_tokens: int,
    batch_size: int,
    device: str,
    dataset: Dataset,
    save_path: str,
) -> None:
    with open(f"{save_path}/metadata.out", "w") as f:
        lines = []
        lines.append(f"epochs: {epochs}\n")
        lines.append(f"new_model: {new_model}\n")
        lines.append(f"dim: {dim}\n")
        lines.append(f"N: {N}\n")
        lines.append(f"max_tokens: {max_tokens}\n")
        lines.append(f"batch_size: {batch_size}\n")
        lines.append(f"device: {device}\n")
        lines.append(f"save_path: {save_path}\n")
        lines.append("\n")
        lines.append("dataset: \n")
        lines.append(dataset.get_description())
        lines.append("\n")
        lines.append("model: \n")
        lines.append(str(model) + "\n")
        lines.append("\n")
        f.writelines(lines)


def train_one_epoch(
    model: TransformerModel,
    optimizer: torch.optim.AdamW,
    dataset: Dataset,
    batch_size: int,
    N: int,
    device: str,
    validate: bool
):
    if validate:
        model.eval()
        torch.set_grad_enabled(False)
    else:
        model.train()

    total_loss = 0.0
    num_batches = 0

    # set up tqdm
    groups, _, _, _ = get_group_by_label(dataset, N)
    total_num_batches = math.ceil(len(groups)/batch_size)
    batch_iterator = get_batches_by_label(dataset, batch_size, N, device)
    desc = "Training" if not validate else "Validating"
    pbar = tqdm(batch_iterator, total=total_num_batches, desc=desc, leave=False)

    for inputs, targets in pbar:
        pred = model(inputs)
        loss = 1 - F.cosine_similarity(pred, targets, dim=-1).mean()

        if not validate:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))

    if validate:
        torch.set_grad_enabled(True)

    return total_loss / num_batches


def validate(
    model: TransformerModel,
    dataset: Dataset,
    batch_size: int,
    device: str,
):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # set up tqdm
    N = 20
    groups, _, _, _ = get_group_by_label(dataset, N)
    total_num_batches = math.ceil(len(groups) / batch_size)
    batch_iterator = get_batches_by_label(dataset, batch_size, N, device)
    pbar = tqdm(batch_iterator, total=total_num_batches, desc="Validating", leave=False)

    for inputs, targets in pbar:
        pred = model(inputs)
        loss = 1 - F.cosine_similarity(pred, targets, dim=-1).mean()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))

    return total_loss / num_batches


# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝


# --- Main ---
def main(
    save_path: Optional[str],
    encoder: str,
    name: Optional[str],
    epochs: int,
    dim: int,
    N: int,
    max_tokens: int,
    batch_size: int,
    train_model: bool = True,
    new_model: bool = True,
    maps: Union[List[int], List[str]] = [],
    intermediate: int = -1,
    only_write_metadata: bool = False,
    validate_all: bool = False,
    debug: bool = False,
):
    if save_path is None:
        save_path = f"models/{name}"

    signal.signal(signal.SIGINT, signal_handler)

    model: TransformerModel
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load CLIP model ---
    if encoder == "lseg":
        backbone = "ViT-B-32"
        clip_model, _, _ = open_clip.create_model_and_transforms(
            backbone, pretrained="openai", force_quick_gelu=True
        )
        clip_model.eval().to(device)
        tokenizer = open_clip.get_tokenizer(backbone)
    elif encoder == "openseg":
        backbone = "ViT-L-14"
        clip_model, _, _ = open_clip.create_model_and_transforms(
            backbone, pretrained="openai", force_quick_gelu=True
        )
        clip_model.eval().to(device)
        tokenizer = open_clip.get_tokenizer(backbone)
    else:
        raise ValueError("Unknown encoder")

    # --- Data ---
    training_dataset = VLMapsDataset(
        "/home/user/hdd/datasets/vlmaps_dataset",
        "/home/user/code/vlmaps/cfg/mpcat40.tsv",
        encoder,
        maps,
        clip_model,
        tokenizer,
        device,
    )

    # --- DEBUG ---
    if only_write_metadata:
        print("Note: debug mode, only write metadata")
        if new_model:
            temp_model = VectorTransformer.load(save_path, dim).to(device)
        else:
            temp_model = LoraAggregator.load(save_path, dim).to(device)
        save_metadata(
            temp_model,
            epochs,
            new_model,
            dim,
            N,
            max_tokens,
            batch_size,
            device,
            training_dataset,
            save_path,
        )
        print("Done.")
        exit()

    # --- Create model ---
    if train_model:
        unused_maps = training_dataset.get_unused_maps()
        if validate_all:
            validation_maps = unused_maps
        else:
            validation_maps = random.sample(unused_maps, 1)
        validation_dataset = VLMapsDataset(
            "/home/user/hdd/datasets/vlmaps_dataset",
            "/home/user/code/vlmaps/cfg/mpcat40.tsv",
            encoder,
            validation_maps,
            clip_model,
            tokenizer,
            device,
        )
        print(f"Training on: {training_dataset.map_strings}")
        print(f"Validating on: {validation_dataset.map_strings}")

        model = train(
            epochs,
            new_model,
            dim,
            N,
            max_tokens,
            batch_size,
            device,
            training_dataset,
            validation_dataset,
            intermediate,
            save_path,
            debug,
            name,
        )

        model.save(save_path)
        save_metadata(
            model,
            epochs,
            new_model,
            dim,
            N,
            max_tokens,
            batch_size,
            device,
            training_dataset,
            save_path,
        )
    else:
        if new_model:
            model = VectorTransformer.load(save_path, dim).to(device)
        else:
            model = LoraAggregator.load(save_path, dim).to(device)

        # --- Evaluate model ---
        eval(model, device, training_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train model from scratch")
    parser.add_argument("--debug", action="store_true", help="Debug run")
    parser.add_argument("--encoder", type=str, help="Encoder", required=True)
    parser.add_argument(
        "--new", action="store_true", help="Train a new model from scratch"
    )
    parser.add_argument("--epochs", type=int, help="Training epochs", default=1000)
    parser.add_argument("--dim", type=int, help="Embedding dimension", default=512)
    parser.add_argument("--n", type=int, help="Batch N", default=20)
    parser.add_argument("--max_tokens", type=int, help="Max tokens", default=512)
    parser.add_argument("--batch", type=int, help="Batch size", default=256)
    parser.add_argument("--path", type=str, help="Save/load path", default=None)
    parser.add_argument("--name", type=str, help="Model name", default=None)
    parser.add_argument("--maps", nargs="+", help="User maps", default=[])
    parser.add_argument(
        "--intermediate", type=int, help="Intermediate save points", default=-1
    )
    parser.add_argument(
        "--only_write_metadata", action="store_true", help="Debug: only write metadata"
    )
    parser.add_argument(
        "--validate_all", action="store_true", help="Validation using all non-training maps"
    )
    args = parser.parse_args()
    parsed_maps: Union[List[int], List[str]] = []
    if args.maps:
        try:
            for item in args.maps:
                parsed_maps.append(int(item))  # type: ignore
        except ValueError:
            parsed_maps = args.maps

    assert args.name is not None or args.path is not None, "You must use --path and/or --name"

    main(
        save_path=args.path,
        encoder=args.encoder,
        name=args.name,
        epochs=args.epochs,
        dim=args.dim,
        N=args.n,
        max_tokens=args.max_tokens,
        batch_size=args.batch,
        train_model=args.train,
        new_model=args.new,
        maps=parsed_maps,
        intermediate=args.intermediate,
        only_write_metadata=args.only_write_metadata,
        validate_all=args.validate_all,
        debug=args.debug
    )
