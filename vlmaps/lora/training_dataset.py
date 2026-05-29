from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import torch
import numpy as np


class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass


def get_batches(dataset: Dataset, batch_size: int, device: str):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)  # shuffle for each epoch

    for start in range(0, len(indices), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        inputs, targets = zip(*(dataset[i] for i in batch_idx))
        yield torch.Tensor(np.array(inputs)).to(device), torch.Tensor(np.array(targets)).to(device)


def get_group_by_label(
    dataset,
    N: int,
    shuffle: bool = True,
    drop_incomplete_groups: bool = False,  # if True, skip groups with < N samples
):
    E: np.ndarray = np.asarray(dataset.embeddings)  # (M, D)
    sem: np.ndarray = np.asarray(dataset.semantics)  # (M,)
    C = dataset.class_embeddings  # (num_classes, D) or mapping
    D = E.shape[1]

    # Map: label -> indices
    unique_labels = np.unique(sem)
    label_to_indices = {int(lbl): np.flatnonzero(sem == lbl) for lbl in unique_labels}

    rng = np.random.default_rng()

    # Build groups (label, idx_chunk up to N)
    groups: List[Tuple[int, np.ndarray]] = []
    for lbl, idxs in label_to_indices.items():
        idxs = idxs.copy()
        if shuffle:
            rng.shuffle(idxs)

        if drop_incomplete_groups:
            # only take full-N chunks
            full = (len(idxs) // N) * N
            for start in range(0, full, N):
                groups.append((lbl, idxs[start : start + N]))
        else:
            # include final partial chunk (size < N)
            for start in range(0, len(idxs), N):
                chunk = idxs[start : start + N]
                if len(chunk) > 0:
                    groups.append((lbl, chunk))
    return groups, rng, E, C

def get_batches_by_label(
    dataset,
    batch_size: int,
    N: int,
    device: str,
    shuffle: bool = True,
    drop_incomplete_groups: bool = False,  # if True, skip groups with < N samples
    drop_last_batch: bool = False,  # if True, drop final short batch
):
    """
    One-pass iterator that yields (inputs, targets) without padding/masking.

    - Groups items by `dataset.semantics`.
    - Makes groups of up to N samples per label.
    - Forms batches of these groups.
    - For each batch, uses N_batch = min(len(group) for group in batch),
      and truncates all groups to N_batch. No masking, no padding.

    Yields:
        inputs:  (B, N_batch, D)  [N_batch can vary per batch]
        targets: (B, D)
    """
    groups, rng, E, C = get_group_by_label(dataset, N, shuffle, drop_incomplete_groups)

    if shuffle:
        rng.shuffle(groups)  # type: ignore

    # Batch the groups
    for start in range(0, len(groups), batch_size):
        batch_groups = groups[start : start + batch_size]
        if len(batch_groups) < batch_size and drop_last_batch:
            continue

        # Determine N_batch (no masks/padding)
        lens = [len(chunk) for _, chunk in batch_groups]
        N_batch = min(lens)  # truncate every group to this length
        if N_batch == 0:
            continue  # shouldn't happen, but safe-guard

        inputs_list: List[np.ndarray] = []
        targets_list: List[np.ndarray] = []

        for lbl, chunk in batch_groups:
            sub = chunk[:N_batch]  # truncate to N_batch
            arr = E[sub]  # (N_batch, D)
            inputs_list.append(arr)
            targets_list.append(C[int(lbl)])

        inputs_np = np.stack(inputs_list, axis=0)  # (B, N_batch, D)
        targets_np = np.stack(targets_list, axis=0)  # (B, D)

        inputs = torch.from_numpy(inputs_np).to(device=device, dtype=torch.float32)
        targets = torch.from_numpy(targets_np).to(device=device, dtype=torch.float32)

        yield inputs, targets
