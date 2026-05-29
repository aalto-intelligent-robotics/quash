import os
import copy
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.datasets import load_iris, make_moons
from sklearn.model_selection import train_test_split
from utils.dialog import asksaveasfilename, askopenfilename
from pathlib import Path


def get_classification(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[int, int, int, int, int]:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    count = np.sum((tp, fp, tn, fn))
    return tp, fp, tn, fn, count


class Result:
    def __init__(self, id) -> None:
        self.id = id

    def get(self, tp: int, fp: int, tn: int, fn: int, count: int) -> None:
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
        self.count = count
        self.get_accuracy()
        self.get_precision()
        self.get_recall()
        self.get_f1()
        self.get_iou()
        self.get_tn_ratio()

    def get_accuracy(self) -> None:
        div = self.tp + self.fp + self.tn + self.fn
        if div != 0:
            self.accuracy = (self.tp + self.tn) / div
        else:
            self.accuracy = 0

    # micro averaged
    def get_precision(self) -> None:
        div = self.tp + self.fp
        if div != 0:
            self.precision = self.tp / div
        else:
            self.precision = 0

    # micro averaged
    def get_recall(self) -> None:
        div = self.tp + self.fn
        if div != 0:
            self.recall = self.tp / div
        else:
            self.recall = 0

    # micro averaged
    def get_f1(self) -> None:
        div = 2 * self.tp + self.fp + self.fn
        if div != 0:
            self.f1 = 2 * self.tp / div
        else:
            self.f1 = 0

    # micro averaged
    def get_iou(self) -> None:
        div = self.tp + self.fp + self.fn
        if div != 0:
            self.iou = self.tp / div
        else:
            self.iou = 0

    def get_tn_ratio(self) -> None:
        self.all_predictions = self.tp + self.fp + self.tn + self.fn
        div = self.all_predictions
        if div != 0:
            self.tn_ratio = self.tn / div
        else:
            self.tn_ratio = 0

    def __str__(self) -> str:
        out = ""
        out += f"tp: {self.tp}\n"
        out += f"fp: {self.fp}\n"
        out += f"tn: {self.tn}\n"
        out += f"fn: {self.fn}\n"
        out += f"count: {self.count}\n"
        out += f"accuracy: {self.accuracy}\n"
        out += f"precision: {self.precision}\n"
        out += f"recall: {self.recall}\n"
        out += f"f1: {self.f1}\n"
        out += f"iou: {self.iou}\n"
        out += f"tn_ratio: {self.tn_ratio}\n"
        return out

class BinaryClassifier(nn.Module):
    def __init__(self, structure: np.ndarray):
        super().__init__()
        self.structure = structure
        input_dim = structure[0]
        hidden_dim = structure[1]
        output_dim = structure[2]
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim),
        )

    def forward(self, x):
        return self.model(x)

    def save(self, save_file: Optional[str] = None):
        file_path = os.path.dirname(os.path.realpath(__file__))
        if save_file is None:
            save_file = asksaveasfilename(
                "Save as",
                f"{file_path}/config/weights",
                [("Weights", "*.pth"), ("All files", "*")],
            )
        torch.save(self.state_dict(), save_file)
        structure_file = Path(save_file).with_suffix(".npy")
        np.save(structure_file, self.structure)

    def load(self, weights_file: Optional[str] = None):
        file_path = os.path.dirname(os.path.realpath(__file__))
        if weights_file is None:
            weights_file = askopenfilename(
                "Load weights",
                f"{file_path}/config/weights",
                [("Weights", "*.pth"), ("All files", "*")],
            )
        structure_file = Path(weights_file).with_suffix(".npy")
        structure = np.load(structure_file)
        self.__init__(structure)

        weights = torch.load(weights_file, weights_only=True)
        print(f"Loaded weights {weights_file}")
        self.load_state_dict(weights)


def init_model(structure: np.ndarray, load: bool):
    # loss metric and optimizer
    model = BinaryClassifier(structure)
    if load:
        model.load()
    # binary
    # loss_fn = nn.BCEWithLogitsLoss()
    # multiclass
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, loss_fn, optimizer


def generate_moons_data(n_samples=1000, train_split=0.1, noise=0.3):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

    # Shuffle the data and labels
    indices = np.random.permutation(
        n_samples
    )  # Generate a random permutation of indices
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X_shuffled).float()
    y_tensor = torch.from_numpy(y_shuffled).float()

    # Split into training and testing sets
    train_size = int(train_split * n_samples)
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

    return X_train, y_train, X_test, y_test


def get_data(encoder: str, train_split: float, map: bool):
    code_path = os.path.dirname(os.path.realpath(__file__)).rstrip("methods/")
    if map:
        y_path = f"{code_path}/training/{encoder}_semantic.npy"
        X_path = f"{code_path}/training/{encoder}_features.npy"
    else:
        y_path = f"{code_path}/training/{encoder}_image_semantic.npy"
        X_path = f"{code_path}/training/{encoder}_image_features.npy"

    y = np.load(y_path)
    X = np.load(X_path)
    data_len = y.shape[0]
    # Shuffle the data and labels
    indices = np.random.permutation(
        data_len
    )  # Generate a random permutation of indices
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X_shuffled).float()
    y_tensor = torch.from_numpy(y_shuffled).float()

    # Split into training and testing sets
    train_size = int(train_split * data_len)
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

    return X_train, y_train.long(), X_test, y_test.long()


def train(
    model: BinaryClassifier,
    loss_fn,
    optimizer,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    device: str,
    n_epochs: int,
    batch_size: int,
):
    model = model.to(device)
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    # prepare model and training parameters
    batches_per_epoch = len(X_train) // batch_size

    best_acc = -np.inf  # init to negative infinity
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []

    # training loop
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        # set model in training mode and run through each batch
        model.train()
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # take a batch
                start = i * batch_size
                X_batch = X_train[start : start + batch_size]
                y_batch = y_train[start : start + batch_size]
                # forward pass
                logits = model(X_batch)
                loss = loss_fn(logits, y_batch.view(-1))
                # y_pred = torch.argmax(logits, dim=1)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # compute and store metrics
                acc = value_function(logits, y_batch)
                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))
                bar.set_postfix(loss=float(loss), acc=float(acc))
        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        ce, acc = eval(model, loss_fn, X_test, y_test)
        test_loss_hist.append(ce)
        test_acc_hist.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        print(
            f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, IoU={acc*100:.1f}%"
        )

    return best_weights, train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist


def value_function(logits, y_test, verbose = False):
    # binary
    # probs = torch.sigmoid(y_pred)
    # preds = (probs > 0.5).float()

    # multiclass
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    tp, fp, tn, fn, count = get_classification(preds.cpu().detach().numpy(), y_test.squeeze().cpu().detach().numpy())
    result = Result(1)
    result.get(tp, fp, tn, fn, count)
    if verbose:
        print("accuracy :", result.accuracy)
        print("precision:", result.precision)
        print("recall   :", result.recall)
        print("f1       :", result.f1)
        print("iou      :", result.iou)
        print("")
    acc = result.iou
    return acc


def eval(model, loss_fn, X_test, y_test, verbose = False):
    # set model in evaluation mode and run through the test set
    model.eval()
    with torch.no_grad():
        all_preds = []
        batch_size = 128

        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                x_batch = X_test[i:i+batch_size]
                y_batch_pred = model(x_batch).squeeze()
                all_preds.append(y_batch_pred)

        y_pred = torch.cat(all_preds)
        ce = loss_fn(y_pred, y_test.view(-1))
        acc = value_function(y_pred, y_test, verbose)
        ce = float(ce)
        acc = float(acc)

    return ce, acc


def visualize(
    model: BinaryClassifier,
    best_weights,
    train_loss_hist,
    test_loss_hist,
    train_acc_hist,
    test_acc_hist,
):
    # Restore best model
    model.load_state_dict(best_weights)

    # Plot the loss and accuracy
    plt.plot(train_loss_hist, label="train")
    plt.plot(test_loss_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.show()

    plt.plot(train_acc_hist, label="train")
    plt.plot(test_acc_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


def main():
    device = "cuda"
    print("1. LSeg")
    print("2. OpenSeg")
    try:
        choice = int(input(": "))
        if choice == 1:
            encoder = "lseg"
            structure = np.array((512, 1024, 42))
        else:
            encoder = "openseg"
            structure = np.array((768, 1024, 42))
    except:
        exit()

    print("1. Train")
    print("2. Eval")
    try:
        choice = int(input(": "))
    except:
        exit()

    train_model = choice == 1
    model, loss_fn, optimizer = init_model(structure, not train_model)

    try:
        train_split = float(input("Train split: "))
        map_data = input("Use map data: ").lower() == "y"
    except:
        train_split = 0.1
        map_data = False
    X_train, y_train, X_test, y_test = get_data(encoder, train_split, map_data)

    if train_model:
        try:
            n_epochs = int(input("Epochs: "))
            batch_size = int(input("Batch size: "))
        except:
            n_epochs = 100
            batch_size = 2048

        best_weights, train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist = (
            train(
                model,
                loss_fn,
                optimizer,
                X_train,
                X_test,
                y_train,
                y_test,
                device,
                n_epochs,
                batch_size,
            )
        )
        visualize(
            model,
            best_weights,
            train_loss_hist,
            test_loss_hist,
            train_acc_hist,
            test_acc_hist,
        )
        model.save()
    else:
        ce, acc = eval(model, loss_fn, X_test, y_test, verbose=True)
        print(f"Validation: Cross-entropy={ce:.2f}, IoU={acc*100:.1f}%")


if __name__ == "__main__":
    main()
