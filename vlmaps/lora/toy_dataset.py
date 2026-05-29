import random
from typing import List, Tuple
import torch
import torch.nn.functional as F
from training_dataset import Dataset
import numpy as np


class ToyDataset(Dataset):
    def __init__(self, device, clip_model, tokenizer) -> None:
        self.device = device
        self.clip_model = clip_model
        self.tokenizer = tokenizer

        self.word_sets = [
                (["leash", "fur", "paw", "bark", "fetch"], "dog"),
                (["whiskers", "purring", "scratching", "litterbox", "feline"], "cat"),
                (["keyboard", "monitor", "processor", "motherboard", "USB"], "computer"),
                (["leaves", "trunk", "branches", "forest", "roots"], "tree"),
                (["feathers", "beak", "wings", "nest", "chirp"], "bird"),
                (["wheels", "engine", "dashboard", "accelerator", "steering"], "car"),
                (["scales", "gills", "aquarium", "swim", "fins"], "fish"),
                (["knife", "stove", "frying", "oven", "cookware"], "kitchen"),
                (["easel", "paint", "canvas", "palette", "brush"], "art"),
                (["helmet", "bicycle", "pedal", "chain", "brake"], "bike"),
                (["spoon", "fork", "plate", "napkin", "dishwasher"], "dining"),
                (["tent", "campfire", "backpack", "sleeping bag", "outdoors"], "camping"),
                (["spacesuit", "rocket", "orbit", "astronaut", "zero gravity"], "space"),
                (["piano", "violin", "orchestra", "composer", "symphony"], "music"),
                (["ball", "goal", "referee", "stadium", "team"], "football"),
            ]

        input_words, label_words = zip(*self.word_sets)
        input_embeddings = []
        target_embeddings = []

        with torch.no_grad():
            for related, label in zip(input_words, label_words):
                related = random.sample(related, len(related))
                tokens = self.tokenizer(related).to(self.device)
                rel_emb = self.clip_model.encode_text(tokens)  # type: ignore
                rel_emb = F.normalize(rel_emb, dim=-1)
                input_embeddings.append(rel_emb.detach().cpu().numpy())

                label_tok = self.tokenizer([label]).to(self.device)
                label_emb = self.clip_model.encode_text(label_tok)  # type: ignore
                label_emb = F.normalize(label_emb, dim=-1)
                target_embeddings.append(label_emb[0].detach().cpu().numpy())
        self.inputs = np.vstack(input_embeddings)
        self.targets = np.vstack(target_embeddings)

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        return self.inputs[idx].reshape(1, -1), self.targets[idx].squeeze()

    def get_description(self) -> str:
        return "Toy dataset"