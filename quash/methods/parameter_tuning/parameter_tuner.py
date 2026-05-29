import optuna
from pathlib import Path
import yaml
from typing import Callable
from methods.common.logging import get_file
from methods.method import Method, Predictor, SynonymType
from methods.predictors import CosineSVM
from methods.common.factory import get_default_embedders
from methods.common.tasktype import TaskType
from methods.common.creators.visualEncoderFactory import EncoderType
import sklearn.svm
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import sklearn.cluster
from enum import Enum


class OptimizationObjective(Enum):
    PARAMS = 0
    SYNONYMS = 1

class ParameterTuner:

    def __init__(
        self,
        encoder: EncoderType,
        classifier: str,
        objective: OptimizationObjective,
        temp_folder: str,
        task: TaskType,
        experiment: str,
        num_trials: int,
        multiprocessing: int,
        no_save: bool,
        start_state: str,
        prompt_engineering: bool,
        synonym_type: SynonymType,
        antonym_ratio: int
    ):
        self.encoder = encoder
        self.classifier = classifier
        self.optimization_objective = objective
        self.temp_folder = temp_folder
        self.task = task
        self.experiment = experiment
        self.num_trials = num_trials
        self.multiprocessing = multiprocessing
        self.no_save = no_save
        self.prompt_engineering = prompt_engineering
        self.synonym_type = synonym_type
        self.antonym_ratio = antonym_ratio

        if no_save:
            self.study = optuna.create_study(direction="maximize")
        else:
            Path(f"parameters/{self.experiment}").mkdir(exist_ok=True, parents=True)
            self.study = optuna.create_study(
                direction="maximize",
                study_name=f"{self.experiment}",
                storage=f"sqlite:///parameters/{self.experiment}/{self.experiment}.db",
                load_if_exists=True,
            )

        if start_state == "default":
            self.study.enqueue_trial(
                {
                    "param_C": 1.0,
                    "param_kernel": "rbf",
                    "param_distance": "squared",
                    "param_degree": 3,
                    "param_gamma": "scale",
                    "param_coef0": 0.0,
                    "param_shrinking": True,
                    "param_probability": False,
                    "param_tol": 0.001,
                    "param_class_weight": None,
                }
            )
        else:
            if Path(start_state).exists():
                try:
                    with open(start_state, "r", encoding="utf-8") as f:
                        load_state = yaml.safe_load(f)
                        init_state = {}
                        init_state["param_C"] = load_state["C"]
                        init_state["param_kernel"] = load_state["kernel"]
                        init_state["param_distance"] = load_state["distance"]
                        init_state["param_degree"] = load_state["degree"]
                        init_state["param_gamma"] = load_state["gamma"]
                        init_state["param_coef0"] = load_state["coef0"]
                        init_state["param_shrinking"] = load_state["shrinking"]
                        init_state["param_probability"] = load_state["probability"]
                        init_state["param_tol"] = load_state["tol"]
                        init_state["param_class_weight"] = load_state["class_weight"]
                        self.study.enqueue_trial(init_state)
                except Exception:
                    pass

    def optimize(self, cost_fun: Callable, **kwargs) -> None:
        if self.optimization_objective == OptimizationObjective.PARAMS:
            self.study.optimize(
                lambda trial: self.objective(trial, cost_fun, **kwargs),
                n_trials=self.num_trials,
                n_jobs=self.multiprocessing,
            )
        elif self.optimization_objective == OptimizationObjective.SYNONYMS:
            self.study.optimize(
                lambda trial: self.objective_synonyms(trial, cost_fun, **kwargs),
                n_trials=self.num_trials,
                n_jobs=self.multiprocessing,
            )
        else:
            raise ValueError("Unknown objective")

        print("Best params:", self.study.best_params)

    def objective(self, trial, cost_fun: Callable, **kwargs):
        predictor = self.get_predictor(trial)
        # Create method instance
        image_embedder, text_embedder = get_default_embedders(
            self.encoder, self.temp_folder, cache=True
        )
        method = Method(task=self.task, encoder=self.encoder, classifier=self.classifier, prompt_engineering=self.prompt_engineering, synonym_type=self.synonym_type, antonym_ratio=self.antonym_ratio)
        method.set_embedders(image_embedder, text_embedder)
        method.set_get_predictor(predictor)
        return cost_fun(method, **kwargs)

    def get_predictor(self, trial):
        if self.classifier == "svm" or self.classifier == "cosine-svm":
            return self.get_svms(trial)
        elif self.classifier == "one-svm":
            return self.get_oneclass_svm(trial)
        elif self.classifier == "dbscan":
            return self.get_dbscan(trial)
        elif self.classifier == "gp":
            return self.get_gpc(trial)
        elif self.classifier == "knn":
            return self.get_knn(trial)
        elif self.classifier == "log":
            return self.get_logistic(trial)
        elif self.classifier == "bayes":
            return self.get_gnb(trial)
        else:
            raise ValueError("Unknown predictor")

    def get_svms(self, trial):
        # Common SVM hyperparameters
        param_C = trial.suggest_float("param_C", 1e-3, 1e3, log=True)
        param_kernel = trial.suggest_categorical(
            "param_kernel", ["linear", "poly", "rbf", "sigmoid", "log", "logn"]
        )

        if param_kernel is ["rbf"]:
            param_distance = trial.suggest_categorical(
                "param_distance", ["linear", "squared"]
            )
        else:
            param_distance = "linear"

        # Parameters that only make sense for certain kernels
        if param_kernel in ["poly", "rbf", "sigmoid", "log", "logn"]:
            param_gamma = trial.suggest_categorical("param_gamma", ["scale", "auto"])
        else:
            param_gamma = "scale"  # Not used

        if param_kernel in ["poly", "logn"]:
            param_degree = trial.suggest_int("param_degree", 2, 5)
            param_coef0 = trial.suggest_float("param_coef0", 0.0, 1.0)
        elif param_kernel == "sigmoid":
            param_coef0 = trial.suggest_float("param_coef0", 0.0, 1.0)
            param_degree = 3  # Not used
        else:
            param_coef0 = 0.0
            param_degree = 3  # Default value

        # Other useful params
        param_shrinking = trial.suggest_categorical("param_shrinking", [True, False])
        param_probability = trial.suggest_categorical(
            "param_probability", [True, False]
        )
        param_tol = trial.suggest_float("param_tol", 1e-5, 1e-1, log=True)
        param_cache_size = 200  # MB, default
        param_class_weight = trial.suggest_categorical(
            "param_class_weight", [None, "balanced"]
        )
        param_verbose = False
        param_max_iter = -1
        param_decision_function_shape = "ovr"
        param_break_ties = False
        param_random_state = 42

        if self.classifier == "svm":
            predictor = sklearn.svm.SVC(
                C=param_C,
                degree=param_degree,
                gamma=param_gamma,
                coef0=param_coef0,
                shrinking=param_shrinking,
                probability=param_probability,
                tol=param_tol,
                cache_size=param_cache_size,
                class_weight=param_class_weight,
                verbose=param_verbose,
                max_iter=param_max_iter,
                decision_function_shape=param_decision_function_shape,
                break_ties=param_break_ties,
                random_state=param_random_state,
            )
        else:
            predictor = CosineSVM("", encoder=self.encoder)
            predictor.init(
                distance=param_distance,
                kernel_type=param_kernel,
                gamma=param_gamma,
                C=param_C,
                coef0=param_coef0,
                degree=param_degree,
                shrinking=param_shrinking,
                probability=param_probability,
                tol=param_tol,
                cache_size=param_cache_size,
                class_weight=param_class_weight,
                verbose=param_verbose,
                max_iter=param_max_iter,
                decision_function_shape=param_decision_function_shape,
                break_ties=param_break_ties,
                random_state=param_random_state,
            )

        return predictor

    def get_oneclass_svm(self, trial):
        param_kernel = trial.suggest_categorical("param_kernel", ["linear", "poly", "rbf", "sigmoid"])
        param_nu = trial.suggest_float("param_nu", 1e-3, 0.5, log=True)
        param_gamma = trial.suggest_categorical("param_gamma", ["scale", "auto"])

        if param_kernel in ["poly", "sigmoid"]:
            param_coef0 = trial.suggest_float("param_coef0", 0.0, 1.0)
        else:
            param_coef0 = 0.0

        if param_kernel == "poly":
            param_degree = trial.suggest_int("param_degree", 2, 5)
        else:
            param_degree = 3  # default value, not used for other kernels

        param_tol = trial.suggest_float("param_tol", 1e-5, 1e-1, log=True)
        param_shrinking = trial.suggest_categorical("param_shrinking", [True, False])
        param_cache_size = 200
        param_verbose = False
        param_max_iter = -1

        clf = sklearn.svm.OneClassSVM(
            kernel=param_kernel,
            nu=param_nu,
            gamma=param_gamma,
            degree=param_degree,
            coef0=param_coef0,
            tol=param_tol,
            shrinking=param_shrinking,
            cache_size=param_cache_size,
            verbose=param_verbose,
            max_iter=param_max_iter
        )
        return clf

    def get_dbscan(self, trial):
        param_eps = trial.suggest_float("param_eps", 0.1, 10.0, log=True)
        param_min_samples = trial.suggest_int("param_min_samples", 3, 20)
        param_metric = trial.suggest_categorical("param_metric", ["euclidean", "manhattan", "chebyshev"])
        param_algorithm = trial.suggest_categorical("param_algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
        param_leaf_size = trial.suggest_int("param_leaf_size", 10, 100)
        param_p = trial.suggest_int("param_p", 1, 5)

        clf = sklearn.cluster.DBSCAN(
            eps=param_eps,
            min_samples=param_min_samples,
            metric=param_metric,
            algorithm=param_algorithm,
            leaf_size=param_leaf_size,
            p=param_p,
            n_jobs=-1
        )
        return clf

    def get_gpc(self, trial):
        param_length_scale = trial.suggest_float("param_length_scale", 0.1, 10.0, log=True)
        param_optimizer = trial.suggest_categorical("param_optimizer", ["fmin_l_bfgs_b", None])
        param_n_restarts = trial.suggest_int("param_n_restarts_optimizer", 0, 5)
        param_max_iter_predict = trial.suggest_int("param_max_iter_predict", 50, 500)
        param_warm_start = trial.suggest_categorical("param_warm_start", [True, False])
        param_copy_X_train = trial.suggest_categorical("param_copy_X_train", [True, False])
        param_multi_class = trial.suggest_categorical("param_multi_class", ["one_vs_rest", "one_vs_one"])

        kernel = RBF(length_scale=param_length_scale)

        clf = GaussianProcessClassifier(
            kernel=kernel,
            optimizer=param_optimizer,
            n_restarts_optimizer=param_n_restarts,
            max_iter_predict=param_max_iter_predict,
            warm_start=param_warm_start,
            copy_X_train=param_copy_X_train,
            multi_class=param_multi_class,
            random_state=42,
            n_jobs=-1
        )
        return clf

    def get_knn(self, trial):
        param_n_neighbors = trial.suggest_int("param_n_neighbors", 1, 30)
        param_weights = trial.suggest_categorical("param_weights", ["uniform", "distance"])
        param_algorithm = trial.suggest_categorical("param_algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
        param_leaf_size = trial.suggest_int("param_leaf_size", 10, 100)
        param_p = trial.suggest_int("param_p", 1, 5)
        param_metric = trial.suggest_categorical("param_metric", ["minkowski", "euclidean", "manhattan"])

        clf = KNeighborsClassifier(
            n_neighbors=param_n_neighbors,
            weights=param_weights,
            algorithm=param_algorithm,
            leaf_size=param_leaf_size,
            p=param_p,
            metric=param_metric,
            n_jobs=-1
        )
        return clf

    def get_logistic(self, trial):
        # Choose solver first
        # param_solver = trial.suggest_categorical("param_solver", [
        #     "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
        # ])
        param_solver = "saga"

        # Match valid penalties per solver
        valid_penalties = {
            "lbfgs": ["l2", None],
            "liblinear": ["l1", "l2"],
            "newton-cg": ["l2", None],
            "newton-cholesky": ["l2", None],
            "sag": ["l2", None],
            "saga": ["elasticnet", "l1", "l2", None],
        }
        allowed_penalties = valid_penalties[param_solver]
        param_penalty = trial.suggest_categorical(
            "param_penalty",
            allowed_penalties
        )

        # C is always applicable unless penalty is None with certain solvers
        param_C = trial.suggest_float("param_C", 1e-3, 1e3, log=True)

        # ElasticNet only if saga + elasticnet
        if param_solver == "saga" and param_penalty == "elasticnet":
            param_l1_ratio = trial.suggest_float("param_l1_ratio", 0.0, 1.0)
        else:
            param_l1_ratio = None

        param_max_iter = trial.suggest_int("param_max_iter", 100, 1000)
        param_fit_intercept = trial.suggest_categorical("param_fit_intercept", [True, False])
        param_multi_class = trial.suggest_categorical("param_multi_class", ["auto", "ovr", "multinomial"])
        param_class_weight = trial.suggest_categorical("param_class_weight", [None, "balanced"])

        clf = sklearn.linear_model.LogisticRegression(
            penalty=param_penalty,
            solver=param_solver,
            C=param_C,
            l1_ratio=param_l1_ratio,
            max_iter=param_max_iter,
            fit_intercept=param_fit_intercept,
            multi_class=param_multi_class,
            class_weight=param_class_weight,
            random_state=42,
            n_jobs=-1
        )
        return clf

    def get_gnb(self, trial):
        param_var_smoothing = trial.suggest_float("param_var_smoothing", 1e-12, 1e-7, log=True)

        clf = GaussianNB(
            var_smoothing=param_var_smoothing
        )
        return clf

    def objective_synonyms(self, trial, cost_fun: Callable, **kwargs):
        # Common SVM hyperparameters
        param_num_synonyms = trial.suggest_int("param_num_synonyms", 3, 40)
        # Create method instance
        image_embedder, text_embedder = get_default_embedders(
            self.encoder, self.temp_folder, cache=True
        )
        method = Method(task=self.task, encoder=self.encoder, classifier=self.classifier, prompt_engineering=self.prompt_engineering, num_of_synonyms=param_num_synonyms)
        method.set_embedders(image_embedder, text_embedder)
        return cost_fun(method, **kwargs)

    def write_results(self) -> None:
        df = self.study.trials_dataframe()
        logfile = get_file(
            f"parameters/{self.experiment}", f"{self.experiment}_log", "csv"
        )
        df.to_csv(logfile, index=False)

        # rename field to match real param dict
        yaml_dict = {}
        for old_key in self.study.best_params.keys():
            new_key = old_key.replace("param_", "")
            val = self.study.best_params[old_key]
            if val is not None:
                yaml_dict[new_key] = val

        # add fields not optimized over
        if "degree" not in yaml_dict:
            yaml_dict["degree"] = 3
        # yaml_dict["cache_size"] = 200
        # yaml_dict["verbose"] = False
        # yaml_dict["max_iter"] = -1
        # yaml_dict["decision_function_shape"] = "ovr"
        # yaml_dict["break_ties"] = False

        yamlfile = get_file(
            f"parameters/{self.experiment}", f"best_params_{self.experiment}", "yaml"
        )
        with open(yamlfile, "w") as file:
            yaml.dump(
                yaml_dict,
                file,
                default_flow_style=False,
                sort_keys=False,
                explicit_start=True,
            )
