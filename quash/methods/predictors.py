from abc import ABC, abstractmethod
import os
import inspect
from pathlib import Path
from typing import Any, Optional, Union, List
import numpy as np
from tqdm import tqdm
import torch
import sklearn
from sklearn import svm
import sklearn.cluster
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    ComplementNB,
    BernoulliNB,
    CategoricalNB,
)
from scipy.linalg import cho_factor, cho_solve
from methods.common.fast_svm import FastSVM
from methods.common.config_reader import ConfigReader
from methods.common.creators.visualEncoderFactory import EncoderType
import methods
from sklearn.metrics.pairwise import (
    cosine_distances,
    euclidean_distances,
    cosine_similarity,
)
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import copy
import tqdm
from itertools import product

# ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗ ██████╗ ██████╗ ███████╗
# ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝
# ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║   ██║██████╔╝███████╗
# ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║   ██║██╔══██╗╚════██║
# ██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ╚██████╔╝██║  ██║███████║
# ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝


class Predictor(ABC):
    def __init__(self, config_path: str, encoder: EncoderType):
        if not config_path:
            config_path = self.get_default_config_path(encoder)
        assert config_path
        config = self.get_config(
            config_path, self.type, encoder, self.get_default_config_path(encoder)
        )
        self.config = ConfigReader(config)

    @property
    @abstractmethod
    def clf(self) -> Any:
        """Classifier that must be implemented by subclasses."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.clf.fit(X, y)

    def predict(self, data: np.ndarray) -> np.ndarray:
        mask = self.clf.predict(data)
        return mask

    @abstractmethod
    def get_default_config_path(self, encoder: EncoderType) -> str:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass

    def get_file_dir(self) -> str:
        return os.path.dirname(inspect.getfile(methods))

    def set_query(self, query: str, query_embedding: np.ndarray) -> None:
        self.query = query
        self.query_embedding = query_embedding

    def set_complement(
        self, complement: List[str], complement_embedding: np.ndarray
    ) -> None:
        self.complement = complement
        self.complement_embedding = complement_embedding

    def get_config(
        self, classifier_config: str, classifier: str, encoder: str, default: str
    ) -> str:
        paths = [
            classifier_config,
            f"~/<path/to/quash>/methods/config/{classifier_config}",
            f"~/<path/to/quash>/methods/config/{classifier}_{classifier_config}",
            f"~/<path/to/quash>/methods/config/{classifier}_{encoder}_{classifier_config}",
        ]
        suffixes = ["", ".yml", ".yaml"]

        config = ""
        for path, suffix in product(paths, suffixes):
            full_path = f"{path}{suffix}"
            if Path(full_path).exists():
                config = full_path
                break
        if config:
            return config
        else:
            return default


class PredictorFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_predictor(
        predictor_type: str, config_path: str, encoder: EncoderType
    ) -> Predictor:
        if predictor_type == "svm":
            return SVM(config_path, encoder)
        elif predictor_type == "cosine-svm":
            return CosineSVM(config_path, encoder)
        elif predictor_type == "one-svm":
            return OneClassSVM(config_path, encoder)
        elif predictor_type == "fast-svm":
            return CustomSVM(config_path, encoder)
        elif predictor_type == "dbscan":
            return DBSCAN(config_path, encoder)
        elif predictor_type == "gp":
            return GaussianProcess(config_path, encoder)
        elif predictor_type == "knn":
            return KNN(config_path, encoder)
        elif predictor_type == "log":
            return Logistic(config_path, encoder)
        elif predictor_type == "bayes":
            return NaiveBayes(config_path, encoder)
        elif predictor_type == "nn":
            return NN(config_path, encoder)
        elif predictor_type == "baseline":
            return Baseline(config_path, encoder)
        elif predictor_type == "baseline_euclidean":
            return BaselineEuclidean(config_path, encoder)
        elif predictor_type == "variance":
            return VarianceWeighted(config_path, encoder)
        elif predictor_type == "normal":
            return NormalDistribution(config_path, encoder)
        elif predictor_type == "dummy":
            return Dummy(config_path, encoder)
        else:
            raise ValueError("Invalid predictor")


class NN(Predictor, nn.Module):
    def __init__(self, config_path: str, encoder: EncoderType):
        Predictor.__init__(self, config_path, encoder)
        nn.Module.__init__(self)
        self.device = "cuda"
        weights = self.config.get_config_def(
            "weights", str, f"methods/config/weights/{encoder}_weights.pth"
        )
        self.load(weights)
        self.weights_path = weights
        self.best_weights = None
        # self.init(np.array((512, 1024, 1)))

    def init(self, structure: np.ndarray):
        input_dim = structure[0]
        hidden_dim = structure[1]
        output_dim = structure[2]
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self.to(self.device)

    def forward(self, x):
        return self.model(x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.run_training(X, y)

    def run_training(self, X: np.ndarray, y: np.ndarray):
        model = self.to(self.device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        n_epochs = 100
        Xt = torch.from_numpy(X).to(device=self.device)
        yt = torch.from_numpy(y).to(device=self.device)

        # prepare model and training parameters
        batches_per_epoch = len(Xt)

        # training loop
        for epoch in range(n_epochs):
            # set model in training mode and run through each batch
            model.train()
            with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
                bar.set_description(f"Epoch {epoch}")
                for i in bar:
                    # forward pass
                    logits = model(Xt).squeeze()
                    loss = loss_fn(logits, yt)
                    # y_pred = torch.argmax(logits, dim=1)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # compute and store metrics
                    bar.set_postfix(loss=float(loss))

        self.best_weights = copy.deepcopy(model.state_dict())  # type: ignore

    def predict(self, data: np.ndarray) -> np.ndarray:
        # self.model.load_state_dict(self.best_weights)
        self.model.eval()
        with torch.no_grad():
            logits = self.forward(torch.from_numpy(data).to(self.device))
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            return preds.cpu().detach().numpy()

    @property
    def clf(self):
        return self._clf

    def load(self, weights_file: str):
        structure = np.load(weights_file.replace("pth", "npy"))
        self.init(structure)

        weights = torch.load(weights_file)
        # print(f"Loaded weights {weights_file}")
        self.load_state_dict(weights)

    def get_description(self) -> str:
        return f"Neural network"

    def get_default_config_path(self, encoder: EncoderType) -> str:
        return self.get_file_dir() + f"/config/nn_{encoder}.yaml"


class CustomSVM(Predictor):
    def __init__(self, config_path: str, encoder: EncoderType):
        super().__init__(config_path, encoder)
        self.initial_gamma = 0.5
        self.C = 1.0
        self.fit_epochs = 1
        self.fit_lr = 0.01
        self.fit_batch_size = 64
        self.predict_batch_size = 512
        learn_gamma = True
        scale_gamma = False

        self._clf = FastSVM(
            initial_gamma=self.initial_gamma,
            C=self.C,
            learn_gamma=learn_gamma,
            scale_gamma=scale_gamma,
        )

    @property
    def clf(self):
        return self._clf

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y[y == 0] = -1
        Xt = torch.tensor(X, dtype=torch.float32).to(self.clf.device)
        yt = torch.tensor(y, dtype=torch.float32).to(self.clf.device)
        self.clf.fit(Xt, yt, self.fit_epochs, self.fit_lr, self.fit_batch_size)

    def predict(self, data: np.ndarray) -> np.ndarray:
        datat = torch.tensor(data, dtype=torch.float32).to(self.clf.device)
        mask = self.clf.predict(datat, self.predict_batch_size)
        mask[mask == -1] = 0
        return mask

    def get_description(self) -> str:
        desc = "FastSVM \n"
        desc += "initial_gamma: " + str(self.initial_gamma) + "\n"
        desc += "C: " + str(self.C) + "\n"
        desc += "fit_epochs: " + str(self.fit_epochs) + "\n"
        desc += "fit_lr: " + str(self.fit_lr) + "\n"
        desc += "fit_batch_size: " + str(self.fit_batch_size) + "\n"
        desc += "predict_batch_size: " + str(self.predict_batch_size) + "\n"
        return desc

    def get_default_config_path(self, encoder: EncoderType) -> str:
        return self.get_file_dir() + "/config/custom_svm.yaml"


class SVM(Predictor):

    def __init__(self, config_path: str, encoder: EncoderType):
        self.type = "svm"
        super().__init__(config_path, encoder)
        self.config_path = config_path
        self.encoder = encoder

        self.pca = None
        self.use_pca = self.config.get_config_def("use_pca", bool, False)
        self.pca_dim = self.config.get_config_def("pca_dim", int, -1)
        self.pca_val = self.config.get_config_def("pca_val", float, -1)

        self._clf = svm.SVC(
            C=self.config.get_config_def("C", float, 1.0),
            kernel=self.config.get_config_def("kernel", str, "rbf"),
            degree=self.config.get_config_def("degree", int, 3),
            gamma=self.config.get_config_def("gamma", str, "scale"),
            coef0=self.config.get_config_def("coef0", float, 0.0),
            shrinking=self.config.get_config_def("shrinking", bool, True),
            probability=self.config.get_config_def("probability", bool, False),
            tol=self.config.get_config_def("tol", float, 0.001),
            cache_size=self.config.get_config_def("cache_size", float, 200),
            class_weight=self.config.get_config_def("class_weight", [float, str], None),
            verbose=self.config.get_config_def("verbose", bool, False),
            max_iter=self.config.get_config_def("max_iter", int, -1),
            decision_function_shape=self.config.get_config_def(
                "decision_function_shape", str, "ovr"
            ),
            break_ties=self.config.get_config_def("break_ties", bool, False),
            random_state=self.config.get_config_def("random_state", int, None),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            if self.pca_dim > 0:
                self.pca = PCA(n_components=self.pca_dim)
            else:
                self.pca = PCA(self.pca_val)
            assert self.pca
            self.pca.fit(X)
        X = self.pca.transform(X)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.use_pca:
            X = self.transform(X)
        y[y == 0] = -1
        self.clf.fit(X, y)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.use_pca:
            data = self.transform(data)
        mask = self.clf.predict(data)
        mask[mask == -1] = 0
        return mask

    @property
    def clf(self):
        return self._clf

    def get_description(self) -> str:
        desc = "Sklearn SVC SVM \n"
        desc += f"encoder: {self.encoder} \n"
        desc += f"config: {self.config_path} \n"
        desc += "-" * 20 + "\n"
        desc += "C: " + str(self.config.get_config_def("C", float, 1.0)) + "\n"
        desc += (
            "kernel: " + str(self.config.get_config_def("kernel", str, "rbf")) + "\n"
        )
        desc += "degree: " + str(self.config.get_config_def("degree", int, 3)) + "\n"
        desc += (
            "gamma: " + str(self.config.get_config_def("gamma", str, "scale")) + "\n"
        )
        desc += "coef0: " + str(self.config.get_config_def("coef0", float, 0.0)) + "\n"
        desc += (
            "shrinking: "
            + str(self.config.get_config_def("shrinking", bool, True))
            + "\n"
        )
        desc += (
            "probability: "
            + str(self.config.get_config_def("probability", bool, False))
            + "\n"
        )
        desc += "tol: " + str(self.config.get_config_def("tol", float, 0.001)) + "\n"
        desc += (
            "cache_size: "
            + str(self.config.get_config_def("cache_size", float, 200))
            + "\n"
        )
        desc += "class_weight: " + (
            str(self.config.get_config_def("class_weight", [float, str], None)) + "\n"
        )
        desc += (
            "verbose: " + str(self.config.get_config_def("verbose", bool, False)) + "\n"
        )
        desc += (
            "max_iter: " + str(self.config.get_config_def("max_iter", int, -1)) + "\n"
        )
        desc += "decision_function_shape: " + (
            str(self.config.get_config_def("decision_function_shape", str, "ovr"))
            + "\n"
        )
        desc += (
            "break_ties: "
            + str(self.config.get_config_def("break_ties", bool, False))
            + "\n"
        )
        desc += (
            "random_state: "
            + str(self.config.get_config_def("random_state", int, None))
            + "\n"
        )
        desc += "-" * 20 + "\n"
        return desc

    def get_default_config_path(self, encoder: EncoderType) -> str:
        path = self.get_file_dir() + f"/config/svm_{encoder}_optimized.yaml"
        if Path(path).exists():
            return path

        path = self.get_file_dir() + f"/config/svm_{encoder}.yaml"
        if Path(path).exists():
            return path

        path = self.get_file_dir() + "/config/svm_default.yaml"
        if Path(path).exists():
            return path

        path = self.get_file_dir() + "/config/svm.yaml"
        if Path(path).exists():
            return path

        raise ValueError("Configuration not found")


class CosineSVM(Predictor):

    def __init__(self, config_path: str, encoder: EncoderType):
        self.type = "cosine-svm"
        super().__init__(config_path, encoder)
        self.config_path = config_path
        self.encoder = encoder
        distance = self.config.get_config_def("distance", str, "squared")
        kernel = self.config.get_config_def("kernel", str, "linear")
        gamma = self.config.get_config_untyped("gamma")
        assert (
            gamma is None
            or gamma == "scale"
            or gamma == "auto"
            or isinstance(gamma, float)
            or isinstance(gamma, int)
        ), "Unknown gamma value"

        self.pca = None
        self.use_pca = self.config.get_config_def("use_pca", bool, False)
        self.pca_dim = self.config.get_config_def("pca_dim", int, -1)
        self.pca_val = self.config.get_config_def("pca_val", float, -1)

        C = self.config.get_config_def("C", float, 1.0)
        coef0 = self.config.get_config_def("coef0", float, 0.0)
        degree = self.config.get_config_def("degree", int, 3)
        shrinking = self.config.get_config_def("shrinking", bool, True)
        probability = self.config.get_config_def("probability", bool, False)
        tol = self.config.get_config_def("tol", float, 0.001)
        cache_size = self.config.get_config_def("cache_size", float, 200)
        class_weight = self.config.get_config_def("class_weight", [float, str], None)
        verbose = self.config.get_config_def("verbose", bool, False)
        max_iter = self.config.get_config_def("max_iter", int, -1)
        decision_function_shape: str = self.config.get_config_def(
            "decision_function_shape", str, "ovr"
        )
        break_ties = self.config.get_config_def("break_ties", bool, False)
        random_state = self.config.get_config_def("random_state", int, None)

        self.init(
            distance,
            kernel,
            gamma,
            C,
            coef0,
            degree,
            shrinking,
            probability,
            tol,
            cache_size,
            class_weight,
            verbose,
            max_iter,
            decision_function_shape,
            break_ties,
            random_state,
        )

    def init(
        self,
        distance: str,
        kernel_type: str,
        gamma: Union[None, str, float, int],
        C: float,
        coef0: float,
        degree: int,
        shrinking: bool,
        probability: bool,
        tol: float,
        cache_size: float,
        class_weight,
        verbose: bool,
        max_iter: int,
        decision_function_shape: str,
        break_ties: bool,
        random_state: Optional[int],
    ):
        self.distance = distance
        if kernel_type == "linear":
            kernel = self.linear_kernel
        elif kernel_type == "log":
            kernel = self.log_kernel
        elif kernel_type == "logn":
            kernel = self.logn_kernel
        elif kernel_type == "poly":
            kernel = self.poly_kernel
        elif kernel_type == "sigmoid":
            kernel = self.sigmoid_kernel
        else:
            kernel = self.rbf_kernel

        self.coef0 = coef0
        self.degree = degree

        if isinstance(gamma, int):
            self.gamma: Union[None, str, float] = float(gamma)
        else:
            self.gamma = None

        self._clf = svm.SVC(
            C=C,
            kernel=kernel,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,  # type: ignore
            break_ties=break_ties,
            random_state=random_state,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            if self.pca_dim > 0:
                self.pca = PCA(n_components=self.pca_dim)
            else:
                self.pca = PCA(self.pca_val)
            assert self.pca is not None
            self.pca.fit(X)
        X = self.pca.transform(X)
        return X

    def linear_kernel(self, X, Y):
        if self.distance == "squared":
            return cosine_similarity(X, Y) ** 2
        else:
            return cosine_similarity(X, Y)

    def get_gamma(self, X, Y) -> float:
        if self.gamma == "scale":
            gamma = 1.0 / (X.shape[0] * np.var(X))
        elif self.gamma == "auto":
            gamma = 1.0 / X.shape[0]
        else:
            if isinstance(self.gamma, float):
                gamma = self.gamma
            else:
                gamma = 1
        return gamma

    def set_gamma(self, X, Y):
        self.gamma = self.get_gamma(X, Y)

    def rbf_kernel(self, X, Y):
        gamma = self.get_gamma(X, Y)
        dist = cosine_distances(X, Y)
        # print(f"gamma: {self.gamma} | {gamma}")
        if self.distance == "squared":
            return np.exp(-gamma * dist**2)  # type: ignore
        else:
            return np.exp(-gamma * dist)  # type: ignore

    def log_kernel(self, X, Y):
        gamma = self.get_gamma(X, Y)
        dist = cosine_distances(X, Y)
        return -np.log((1 - self.coef0) + gamma * dist)

    def logn_kernel(self, X, Y):
        gamma = self.get_gamma(X, Y)
        dist = cosine_distances(X, Y)
        return -np.emath.logn(self.degree, (1 - self.coef0) + gamma * dist)  # type: ignore

    def sigmoid_kernel(self, X, Y):
        gamma = self.get_gamma(X, Y)
        dist = cosine_similarity(X, Y)
        return np.tanh(gamma * dist + self.coef0)

    def poly_kernel(self, X, Y):
        gamma = self.get_gamma(X, Y)
        dist = cosine_similarity(X, Y)
        return (gamma * dist + self.coef0) ** self.degree

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.use_pca:
            X = self.transform(X)
        y[y == 0] = -1
        self.clf.fit(X, y)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.use_pca:
            data = self.transform(data)
        mask = self.clf.predict(data)
        mask[mask == -1] = 0
        return mask

    @property
    def clf(self):
        return self._clf

    def get_description(self) -> str:
        desc = "Cosine SVM \n"
        desc += f"encoder: {self.encoder} \n"
        desc += f"config: {self.config_path} \n"
        desc += "-" * 20 + "\n"
        desc += "C: " + str(self.config.get_config_def("C", float, 1.0)) + "\n"
        desc += (
            "kernel: " + str(self.config.get_config_def("kernel", str, "rbf")) + "\n"
        )
        desc += (
            "distance: "
            + str(self.config.get_config_def("distance", str, "linear"))
            + "\n"
        )
        desc += "degree: " + str(self.config.get_config_def("degree", int, 3)) + "\n"
        desc += (
            "gamma: " + str(self.config.get_config_def("gamma", str, "scale")) + "\n"
        )
        desc += "coef0: " + str(self.config.get_config_def("coef0", float, 0.0)) + "\n"
        desc += (
            "shrinking: "
            + str(self.config.get_config_def("shrinking", bool, True))
            + "\n"
        )
        desc += (
            "probability: "
            + str(self.config.get_config_def("probability", bool, False))
            + "\n"
        )
        desc += "tol: " + str(self.config.get_config_def("tol", float, 0.001)) + "\n"
        desc += (
            "cache_size: "
            + str(self.config.get_config_def("cache_size", float, 200))
            + "\n"
        )
        desc += "class_weight: " + (
            str(self.config.get_config_def("class_weight", [float, str], None)) + "\n"
        )
        desc += (
            "verbose: " + str(self.config.get_config_def("verbose", bool, False)) + "\n"
        )
        desc += (
            "max_iter: " + str(self.config.get_config_def("max_iter", int, -1)) + "\n"
        )
        desc += "decision_function_shape: " + (
            str(self.config.get_config_def("decision_function_shape", str, "ovr"))
            + "\n"
        )
        desc += (
            "break_ties: "
            + str(self.config.get_config_def("break_ties", bool, False))
            + "\n"
        )
        desc += (
            "random_state: "
            + str(self.config.get_config_def("random_state", int, None))
            + "\n"
        )
        desc += (
            "use_pca: " + str(self.config.get_config_def("use_pca", bool, False)) + "\n"
        )
        desc += "pca_dim: " + str(self.config.get_config_def("pca_dim", int, -1)) + "\n"
        desc += (
            "pca_val: " + str(self.config.get_config_def("pca_val", float, -1)) + "\n"
        )
        desc += "-" * 20 + "\n"
        return desc

    def get_default_config_path(self, encoder: EncoderType) -> str:
        path = self.get_file_dir() + f"/config/cosine-svm_{encoder}_optimized.yaml"
        if Path(path).exists():
            return path

        path = self.get_file_dir() + f"/config/cosine-svm_{encoder}.yaml"
        if Path(path).exists():
            return path

        path = self.get_file_dir() + "/config/cosine-svm_default.yaml"
        if Path(path).exists():
            return path

        path = self.get_file_dir() + "/config/cosine-svm.yaml"
        if Path(path).exists():
            return path

        raise ValueError("Configuration not found")


class OneClassSVM(Predictor):

    def __init__(self, config_path: str, encoder: EncoderType):
        self.type = "one-svm"
        super().__init__(config_path, encoder)
        self.config_path = config_path
        self.encoder = encoder

        self._clf = svm.OneClassSVM(
            nu=self.config.get_config_def("nu", float, 0.5),
            kernel=self.config.get_config_def("kernel", str, "rbf"),
            degree=self.config.get_config_def("degree", int, 3),
            gamma=self.config.get_config_def("gamma", str, "scale"),
            coef0=self.config.get_config_def("coef0", float, 0.0),
            shrinking=self.config.get_config_def("shrinking", bool, True),
            tol=self.config.get_config_def("tol", float, 0.001),
            cache_size=self.config.get_config_def("cache_size", float, 200),
            verbose=self.config.get_config_def("verbose", bool, False),
            max_iter=self.config.get_config_def("max_iter", int, -1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y[y == 0] = -1
        self.clf.fit(X, y)

    def predict(self, data: np.ndarray) -> np.ndarray:
        mask = self.clf.predict(data)
        mask[mask == -1] = 0
        return mask

    @property
    def clf(self):
        return self._clf

    def get_description(self) -> str:
        desc = "Sklearn OneClassSVM \n"
        desc += f"encoder: {self.encoder} \n"
        desc += f"config: {self.config_path} \n"
        desc += "-" * 20 + "\n"
        desc += (
            "kernel: " + str(self.config.get_config_def("kernel", str, "rbf")) + "\n"
        )
        desc += "degree: " + str(self.config.get_config_def("degree", int, 3)) + "\n"
        desc += (
            "gamma: " + str(self.config.get_config_def("gamma", str, "scale")) + "\n"
        )
        desc += "coef0: " + str(self.config.get_config_def("coef0", float, 0.0)) + "\n"
        desc += (
            "shrinking: "
            + str(self.config.get_config_def("shrinking", bool, True))
            + "\n"
        )
        desc += "tol: " + str(self.config.get_config_def("tol", float, 0.001)) + "\n"
        desc += (
            "cache_size: "
            + str(self.config.get_config_def("cache_size", float, 200))
            + "\n"
        )
        desc += (
            "verbose: " + str(self.config.get_config_def("verbose", bool, False)) + "\n"
        )
        desc += (
            "max_iter: " + str(self.config.get_config_def("max_iter", int, -1)) + "\n"
        )
        desc += "-" * 20 + "\n"
        return desc

    def get_default_config_path(self, encoder: EncoderType) -> str:
        path = self.get_file_dir() + f"/config/one-svm_{encoder}_optimized.yaml"
        if Path(path).exists():
            return path

        path = self.get_file_dir() + f"/config/one-svm_{encoder}.yaml"
        if Path(path).exists():
            return path

        path = self.get_file_dir() + "/config/one-svm_default.yaml"
        if Path(path).exists():
            return path

        path = self.get_file_dir() + "/config/one-svm.yaml"
        if Path(path).exists():
            return path

        raise ValueError("Configuration not found")


class Dummy(Predictor):
    def __init__(self, _: str, __: EncoderType):
        self.type = "dummy"
        self._clf = None

    @property
    def clf(self):
        return self._clf

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, data: np.ndarray) -> np.ndarray:
        mask = np.zeros((data.shape[0]))
        return mask

    def get_description(self) -> str:
        return "Dummy predictor"

    def get_default_config_path(self, encoder: EncoderType) -> str:
        return ""


class DBSCAN(Predictor):

    def __init__(self, config_path: str, encoder: EncoderType):
        self.type = "dbscan"
        super().__init__(config_path, encoder)
        self.config_path = config_path
        self._clf = sklearn.cluster.DBSCAN(
            eps=self.config.get_config_def("eps", float, 0.5),
            min_samples=self.config.get_config_def("min_samples", int, 5),
            metric=self.config.get_config_def("metric", str, "euclidean"),
            algorithm=self.config.get_config_def("algorithm", str, "auto"),
            leaf_size=self.config.get_config_def("leaf_size", int, 30),
            p=self.config.get_config_def("p", int, None),
            n_jobs=self.config.get_config_def("n_jobs", int, None),
        )

    @property
    def clf(self):
        return self._clf

    def predict(self, data: np.ndarray) -> np.ndarray:
        mask = self.clf.fit_predict(data)
        neg = mask == -1
        mask[neg] = 0
        return mask

    def get_description(self) -> str:
        return f"Sklearn DBSCAN with config: {self.config_path}"

    def get_default_config_path(self, encoder: EncoderType) -> str:
        return self.get_file_dir() + "/config/dbscan.yaml"


class GaussianProcess(Predictor):

    def __init__(self, config_path: str, encoder: EncoderType):
        self.type = "gp"
        super().__init__(config_path, encoder)
        self.config_path = config_path
        self.encoder = encoder

        kernel = RBF(
            length_scale=self.config.get_config_def("length_scale", float, 1.0)
        )

        self._clf = GaussianProcessClassifier(
            kernel=kernel,
            optimizer=self.config.get_config_def("optimizer", str, "fmin_l_bfgs_b"),
            n_restarts_optimizer=self.config.get_config_def(
                "n_restarts_optimizer", int, 0
            ),
            max_iter_predict=self.config.get_config_def("max_iter_predict", int, 100),
            warm_start=self.config.get_config_def("warm_start", bool, False),
            copy_X_train=self.config.get_config_def("copy_X_train", bool, True),
            random_state=self.config.get_config_def("random_state", int, None),
            multi_class=self.config.get_config_def("multi_class", str, "one_vs_rest"),
            n_jobs=self.config.get_config_def("n_jobs", int, None),
        )

    @property
    def clf(self):
        return self._clf

    def get_description(self) -> str:
        return f"Sklearn GaussianProcessClassifier with config: {self.config_path}"

    def get_default_config_path(self, encoder: EncoderType) -> str:
        return self.get_file_dir() + "/config/gp.yaml"


class KNN(Predictor):

    def __init__(self, config_path: str, encoder: EncoderType):
        self.type = "knn"
        super().__init__(config_path, encoder)
        self.config_path = config_path
        self.encoder = encoder

        self._clf = KNeighborsClassifier(
            n_neighbors=self.config.get_config_def("n_neighbors", int, 5),
            weights=self.config.get_config_def("weights", str, "uniform"),
            algorithm=self.config.get_config_def("algorithm", str, "auto"),
            leaf_size=self.config.get_config_def("leaf_size", int, 30),
            p=self.config.get_config_def("p", int, 2),
            metric=self.config.get_config_def("metric", str, "minkowski"),
            n_jobs=self.config.get_config_def("n_jobs", int, None),
        )

    @property
    def clf(self):
        return self._clf

    def get_description(self) -> str:
        return f"Sklearn KNeighborsClassifier with config: {self.config_path}"

    def get_default_config_path(self, encoder: EncoderType) -> str:
        return self.get_file_dir() + "/config/knn.yaml"


class Logistic(Predictor):

    def __init__(self, config_path: str, encoder: EncoderType):
        self.type = "log"
        super().__init__(config_path, encoder)
        self.config_path = config_path
        self.encoder = encoder

        self._clf = LogisticRegression(
            penalty=self.config.get_config_def("penalty", str, "l2"),
            dual=self.config.get_config_def("dual", bool, False),
            C=self.config.get_config_def("C", float, 1.0),
            fit_intercept=self.config.get_config_def("fit_intercept", bool, True),
            solver=self.config.get_config_def("solver", str, "lbfgs"),
            max_iter=self.config.get_config_def("max_iter", int, 100),
            multi_class=self.config.get_config_def("multi_class", str, "auto"),
            class_weight=self.config.get_config_def("class_weight", [str, dict], None),
            n_jobs=self.config.get_config_def("n_jobs", int, None),
            random_state=self.config.get_config_def("random_state", int, None),
            verbose=self.config.get_config_def("verbose", int, 0),
        )

    @property
    def clf(self):
        return self._clf

    def get_description(self) -> str:
        return f"Sklearn LogisticRegression with config: {self.config_path}"

    def get_default_config_path(self, encoder: EncoderType) -> str:
        return self.get_file_dir() + "/config/log.yaml"


class NaiveBayes(Predictor):

    def __init__(self, config_path: str, encoder: EncoderType):
        self.type = "bayes"
        super().__init__(config_path, encoder)
        self.config_path = config_path
        self.encoder = encoder

        self.clf_type = self.config.get_config_def("type", str, "gaussian")

        if self.clf_type == "gaussian":
            self._clf = GaussianNB(
                var_smoothing=self.config.get_config_def("var_smoothing", float, 1e-9)
            )
        elif self.clf_type == "multinomial":
            self._clf = MultinomialNB(
                alpha=self.config.get_config_def("alpha", float, 1.0)
            )
        elif self.clf_type == "complement":
            self._clf = ComplementNB(
                alpha=self.config.get_config_def("alpha", float, 1.0)
            )
        elif self.clf_type == "bernoulli":
            self._clf = BernoulliNB(
                alpha=self.config.get_config_def("alpha", float, 1.0)
            )
        elif self.clf_type == "categorical":
            self._clf = CategoricalNB(
                alpha=self.config.get_config_def("alpha", float, 1.0)
            )
        else:
            raise ValueError(f"Unknown Naive Bayes type: {self.clf_type}")

    @property
    def clf(self):
        return self._clf

    def get_description(self) -> str:
        return f"Sklearn {self.clf_type} naive bayes with config: {self.config_path}"

    def get_default_config_path(self, encoder: EncoderType) -> str:
        return self.get_file_dir() + "/config/bayes.yaml"


class Baseline(Predictor):

    def __init__(self, config_path: str, encoder: EncoderType):
        self.type = "baseline"
        super().__init__(config_path, encoder)
        self._clf = None
        self.use_all = self.config.get_config_def("all", bool, False)

    @property
    def clf(self):
        return self._clf

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.use_all:
            return self.fit_all(X, y)
        else:
            return self.fit_one(X, y)

    def fit_one(self, X: np.ndarray, y: np.ndarray) -> None:
        self.positive_idx = 0
        self.examples = np.vstack((self.query_embedding, self.complement_embedding))

    def fit_all(self, X: np.ndarray, y: np.ndarray) -> None:
        self.positive_idx = np.argwhere(y == 1).squeeze()
        self.examples = X

    def predict(self, data: np.ndarray) -> np.ndarray:
        sims = cosine_similarity(data, self.examples)
        max_sim = np.argmax(sims, axis=1)
        predictions = np.isin(max_sim, self.positive_idx)
        return predictions

    def get_description(self) -> str:
        return "Baseline predictor"

    def get_default_config_path(self, encoder: EncoderType) -> str:
        return self.get_file_dir() + "/config/baseline.yaml"


class BaselineEuclidean(Predictor):

    def __init__(self, config_path: str, encoder: EncoderType):
        self.type = "baseline-euclidean"
        super().__init__(config_path, encoder)
        self._clf = None
        self.use_all = self.config.get_config_def("all", bool, False)

    @property
    def clf(self):
        return self._clf

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.use_all:
            return self.fit_all(X, y)
        else:
            return self.fit_one(X, y)

    def fit_one(self, X: np.ndarray, y: np.ndarray) -> None:
        self.positive_idx = 0
        self.examples = np.vstack((self.query_embedding, self.complement_embedding))

    def fit_all(self, X: np.ndarray, y: np.ndarray) -> None:
        self.positive_idx = np.argwhere(y == 1).squeeze()
        self.examples = X

    def predict(self, data: np.ndarray) -> np.ndarray:
        sims = euclidean_distances(data, self.examples)
        max_sim = np.argmax(sims, axis=1)
        predictions = np.isin(max_sim, self.positive_idx)
        # metric is now distance, not similarity, so opposite is true
        predictions = np.logical_not(predictions)
        return predictions

    def get_description(self) -> str:
        return "Baseline predictor"

    def get_default_config_path(self, encoder: EncoderType) -> str:
        return self.get_file_dir() + "/config/baseline_euclidean.yaml"


class VarianceWeighted(Predictor):

    def __init__(self, config_path: str, encoder: EncoderType):
        self.type = "variance"
        super().__init__(config_path, encoder)
        self._clf = None
        self.use_all = self.config.get_config_def("all", bool, False)

    @property
    def clf(self):
        return self._clf

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.use_all:
            return self.fit_all(X, y)
        else:
            return self.fit_one(X, y)

    def fit_one(self, X: np.ndarray, y: np.ndarray) -> None:
        synonyms = X[y == 1]
        self.variance = synonyms.var(axis=0)
        self.positive_idx = 0
        self.examples = np.vstack((self.query_embedding, self.complement_embedding))

    def fit_all(self, X: np.ndarray, y: np.ndarray) -> None:
        synonyms = X[y == 1]
        self.variance = synonyms.var(axis=0)
        self.positive_idx = np.argwhere(y == 1).squeeze()
        self.examples = X

    def predict(self, data: np.ndarray) -> np.ndarray:
        sims = VarianceWeighted.pairwise_weighted_cosine_similarity(
            data, self.examples, self.variance
        )
        max_sim = np.argmax(sims, axis=1)
        predictions = np.isin(max_sim, self.positive_idx)
        return predictions

    def get_description(self) -> str:
        return "Variance weighted cosine distance predictor"

    def get_default_config_path(self, encoder: EncoderType) -> str:
        return self.get_file_dir() + "/config/variance.yaml"

    # def weighed_dot(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    #     w = nonlinearity(w)
    #     return float(np.sum(np.multiply(np.multiply(a, b), w)))

    # def cosine_distance_var(a: np.ndarray, b: np.ndarray, var: np.ndarray) -> float:
    #     inv_var = 1 - var
    #     div = float(np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2))
    #     if div == 0:
    #         div = 1e-6
    #     d = weighed_dot(a, b, inv_var) / div
    #     if d > 1:
    #         d = 1.0
    #     if d < -1:
    #         d = -1.0
    #     return d

    # @staticmethod
    # def weighed_dot(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    #     return float(np.sum(np.multiply(np.multiply(a, b), w)))

    # @staticmethod
    # def cosine_distance_var(a: np.ndarray, b: np.ndarray, var: np.ndarray) -> float:
    #     inv_var = np.abs(max(np.max(var), 1) - var)
    #     dim = float(len(inv_var))
    #     w = inv_var / np.sum(inv_var) * dim
    #     w = np.clip(w, 0, 1)
    #     a_mod = np.multiply(a, w)
    #     b_mod = np.multiply(b, w)
    #     div = float(np.linalg.norm(a_mod, ord=2) * np.linalg.norm(b_mod, ord=2))
    #     if div == 0:
    #         div = 1e-6
    #     d = VarianceWeighted.weighed_dot(a, b, w) / div
    #     if d > 1:
    #         d = 1.0
    #     if d < -1:
    #         d = -1.0
    #     return d

    @staticmethod
    def pairwise_weighted_cosine_distance(
        A: np.ndarray, B: np.ndarray, var: np.ndarray
    ) -> np.ndarray:
        return 1 - VarianceWeighted.pairwise_weighted_cosine_similarity(A, B, var)

    @staticmethod
    def pairwise_weighted_cosine_similarity(
        A: np.ndarray, B: np.ndarray, var: np.ndarray
    ) -> np.ndarray:
        # Invert variance for weighting
        var = np.clip(var, 1e-8, None)
        w = 1.0 / var
        w = w / np.sum(w) * len(w)
        # w = np.clip(w, 0.0, 1.0)

        # Apply weights to features
        A_w = A * w
        B_w = B * w

        # Compute dot products
        sim_matrix = A_w @ B_w.T

        # Compute norms
        norm_A = np.linalg.norm(A_w, axis=1, keepdims=True)  # shape (n, 1)
        norm_B = np.linalg.norm(B_w, axis=1, keepdims=True).T  # shape (1, m)

        denom = norm_A * norm_B
        denom = np.clip(denom, 1e-8, None)

        sim_matrix = sim_matrix / denom
        sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
        return sim_matrix


class NormalDistribution(Predictor):

    def __init__(self, config_path: str, encoder: EncoderType):
        self.type = "normal"
        super().__init__(config_path, encoder)
        self._clf = None

    @property
    def clf(self):
        return self._clf

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        synonyms = X[y == 1]
        antonyms = X[y == 0]
        self.synonyms_mean = synonyms.mean(axis=0)
        self.synonyms_covariance = np.cov(synonyms.T)
        self.antonyms_mean = antonyms.mean(axis=0)
        self.antonyms_covariance = np.cov(antonyms.T)

        if not np.all(np.linalg.eigvals(self.synonyms_covariance) > 0):
            var = synonyms.var(axis=0)
            self.synonyms_covariance = np.diag(var)

        if not np.all(np.linalg.eigvals(self.antonyms_covariance) > 0):
            var = antonyms.var(axis=0)
            self.antonyms_covariance = np.diag(var)

        self.synonyms_inv_cov = np.linalg.inv(self.synonyms_covariance)
        self.antonyms_inv_cov = np.linalg.inv(self.antonyms_covariance)

    def mahalanobis(
        self, data: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray
    ) -> np.ndarray:
        diffs = data - mean  # (k, N)
        # Mahalanobis distance for each row: sqrt((x - μ)^T Σ⁻¹ (x - μ))
        dists = np.sqrt(np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs))
        return dists  # shape: (k,)

    def mahalanobis_cholesky(self, data, mean, cov):
        # Cholesky decomposition of covariance
        c_factor, lower = cho_factor(cov, lower=True, check_finite=False)
        diffs = data - mean
        # Solve Σ^{-1/2} (x - μ)
        whitened = cho_solve((c_factor, lower), diffs.T, check_finite=False).T
        return np.linalg.norm(whitened, axis=1)

    def predict(self, data: np.ndarray) -> np.ndarray:
        d_syn = self.mahalanobis(data, self.synonyms_mean, self.synonyms_inv_cov)
        d_ant = self.mahalanobis(data, self.antonyms_mean, self.antonyms_inv_cov)
        # d_syn = self.mahalanobis_cholesky(data, self.synonyms_mean, self.synonyms_covariance)
        # d_ant = self.mahalanobis_cholesky(data, self.antonyms_mean, self.antonyms_covariance)
        return (d_syn < d_ant).astype(int)

    def get_description(self) -> str:
        return "Normal distribution predictor"

    def get_default_config_path(self, encoder: EncoderType) -> str:
        return self.get_file_dir() + "/config/normal.yaml"
