import joblib
import json
import math
import time
import numpy as np
import optuna
import os
import shutil
import sklearn
import h5py
import multiprocessing
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flaml.default import RandomForestClassifier as ZeroShotRandomForestClassifier
from .utils import BinaryClassifierSamplingUtils as SamplingUtils
from .utils import InputUtils
from .optimizers import PCADimensionsOptimizerForBinaryClassification


NUM_CPU = max(1, int(multiprocessing.cpu_count() / 2))


class BaseRandomForestBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        pca: bool = False,
        num_splits: int = 3,
        test_size: float = 0.25,
        num_trials: int = 50,
        timeout: int = 600,
        random_state: int = 42,
    ):
        self.pca = pca
        self.random_state = random_state
        self.num_splits = num_splits
        self.test_size = test_size
        self.num_trials = num_trials
        self.timeout = timeout
        self.mean_score_ = None

    def _suggest_param_search(self, hyperparams, n_samples, n_features, test_size):
        def_n_estimators = hyperparams["n_estimators"]
        min_n_estimators = max(10, int(def_n_estimators) - 100)
        max_n_estimators = min(1000, int(def_n_estimators) + 100)
        def_max_features = hyperparams["max_features"]
        if (
            def_max_features == "auto"
            or def_max_features == "sqrt"
            or def_max_features == "log2"
        ):
            max_features_range = [def_max_features]
        else:
            if def_max_features is None:
                def_max_features = 1.0
            elif def_max_features > 1:
                def_max_features = def_max_features / n_features
            else:
                pass
            min_max_features = max(0.05, def_max_features - 0.2)
            max_max_features = min(1.0, def_max_features + 0.2)
            max_features_range = sorted([min_max_features, max_max_features])
            if min_max_features == max_max_features:
                min_max_features = min_max_features - 0.01
        if hyperparams["max_leaf_nodes"] is None:
            max_leaf_nodes_range = [None]
        else:
            def_max_leaf_nodes = hyperparams["max_leaf_nodes"]
            def_max_leaf_nodes_log2 = (
                math.log2(def_max_leaf_nodes) if def_max_leaf_nodes > 0 else 0
            )
            min_max_leaf_nodes_log2 = max(4, def_max_leaf_nodes_log2 - 1)
            max_max_leaf_nodes_log2 = min(10, def_max_leaf_nodes_log2 + 1)
            min_max_leaf_nodes = int(np.round(2**min_max_leaf_nodes_log2, 0))
            max_max_leaf_nodes = int(np.round(2**max_max_leaf_nodes_log2, 0))
            min_max_leaf_nodes = min(
                min_max_leaf_nodes, int(n_samples * (1 - test_size))
            )
            max_max_leaf_nodes = max(
                max_max_leaf_nodes,
                max(int(n_samples * (1 - test_size)), def_max_leaf_nodes),
            )
            max_leaf_nodes_range = sorted([min_max_leaf_nodes, max_max_leaf_nodes])
            if min_max_leaf_nodes == max_max_leaf_nodes:
                min_max_leaf_nodes = min_max_leaf_nodes - 1
        if "criterion" in hyperparams:
            criterion = hyperparams["criterion"]
        else:
            criterion = "gini"
        if min_n_estimators == max_n_estimators:
            max_n_estimators = min_n_estimators + 1
        param = {
            "n_estimators": sorted([min_n_estimators, max_n_estimators]),
            "max_features": max_features_range,
            "max_leaf_nodes": max_leaf_nodes_range,
            "criterion": [criterion],
            "class_weight": ["balanced_subsample"],
        }
        return param

    def _objective(self, trial, X, y, param):
        param_ = {
            "n_estimators": trial.suggest_int(
                "n_estimators", param["n_estimators"][0], param["n_estimators"][1]
            )
        }
        if len(param["max_features"]) == 1:
            param_["max_features"] = trial.suggest_categorical(
                "max_features", param["max_features"]
            )
        else:
            param_["max_features"] = trial.suggest_float(
                "max_features", param["max_features"][0], param["max_features"][1]
            )
        if len(param["max_leaf_nodes"]) == 1:
            param_["max_leaf_nodes"] = trial.suggest_categorical(
                "max_leaf_nodes", param["max_leaf_nodes"]
            )
        else:
            param_["max_leaf_nodes"] = trial.suggest_int(
                "max_leaf_nodes",
                param["max_leaf_nodes"][0],
                param["max_leaf_nodes"][1],
                log=True,
            )
        param_["criterion"] = trial.suggest_categorical("criterion", param["criterion"])
        param = param_
        param["n_jobs"] = NUM_CPU
        scores = []
        for _ in range(self.num_splits):
            train_x, valid_x, train_y, valid_y = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )
            model = RandomForestClassifier(**param)
            model.fit(train_x, train_y)
            preds = model.predict_proba(valid_x)[:, 1]
            score = sklearn.metrics.roc_auc_score(valid_y, preds)
            scores += [float(score)]
        return np.mean(scores)

    def _suggest_best_params(self, X, y, test_size):
        zero_shot_cv = ZeroShotRandomForestClassifier()
        hyperparams = zero_shot_cv.suggest_hyperparams(X, y)[0]
        if hyperparams == {}:
            hyperparams = {
                "n_estimators": 100,
                "max_features": "sqrt",
                "max_leaf_nodes": None,
                "criterion": "gini",
            }
        hyperparams["class_weight"] = "balanced_subsample"
        print("Suggested zero-shot hyperparameters:", hyperparams)
        n_samples, n_features = X.shape
        hyperparam_search = self._suggest_param_search(
            hyperparams, n_samples, n_features, test_size
        )

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction="maximize",
            study_name=None,
            sampler=sampler,
        )

        study.enqueue_trial(hyperparams)

        print("Fitting...")
        best_score = -np.inf
        trials_without_improvement = 0
        improvement_threshold = 0.01
        patience = max(5, self.num_trials // 5)
        early_stopping = False
        baseline_score_for_patience = best_score

        def objective_with_custom_early_stop(trial):
            nonlocal \
                best_score, \
                trials_without_improvement, \
                early_stopping, \
                baseline_score_for_patience
            if early_stopping:
                print("Skipping trial due to early stopping criteria.")
                raise optuna.exceptions.TrialPruned()
            score = self._objective(trial, X, y, hyperparam_search)
            if score > best_score:
                best_score = score
            if score > baseline_score_for_patience + improvement_threshold:
                trials_without_improvement = 0
                baseline_score_for_patience = score
            else:
                trials_without_improvement += 1
            if trials_without_improvement >= patience:
                early_stopping = True
                print(
                    f"Early stopping: No significant improvement in the last {patience} trials."
                )
                raise optuna.exceptions.TrialPruned()
            return score

        study.optimize(
            objective_with_custom_early_stop,
            n_trials=self.num_trials,
            timeout=self.timeout,
        )

        results = {
            "best_params": study.best_params,
            "best_value": study.best_value,
        }

        return results
    
    def _fit_pca(self, X, y):
        best_params = PCADimensionsOptimizerForBinaryClassification(
                            num_splits = min(self.num_splits, 3),
                            test_size = self.test_size,
                            random_state = self.random_state,
                            num_trials = min(5, self.num_trials),
                            timeout = min(60, self.timeout),
                            max_positive_proportion = self.max_positive_proportion
                        ).get_best_params(X, y)["best_params"]
        print("Working on the PCA")
        n_components = best_params["n_components"]
        reducer = PCA(n_components=n_components)
        reducer.fit(X)
        self.reducer_ = reducer
        X = reducer.transform(X)
        results = self._suggest_best_params(X, y, self.test_size)
        hyperparams = results["best_params"]
        score = results["best_value"]
        hyperparams["n_jobs"] = NUM_CPU
        hyperparams["random_state"] = self.random_state
        hyperparams["class_weight"] = "balanced_subsample"
        print(
            f"Best hyperparameters: {hyperparams}, Inner hyperparameter AUROC: {score}"
        )
        scores = []
        y_cal = []
        probs_cal = []
        for r in range(self.num_splits):
            train_x, valid_x, train_y, valid_y = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state + r,
                stratify=y,
            )
            model_cv = RandomForestClassifier(**hyperparams)
            model_cv.fit(train_x, train_y)
            valid_y_hat = model_cv.predict_proba(valid_x)[:, 1]
            fpr, tpr, _ = roc_curve(valid_y, valid_y_hat)
            auroc = auc(fpr, tpr)
            scores.append(auroc)
            print(f"Internal AUROC CV-{r}: {auroc}")
            y_cal += list(valid_y)
            probs_cal += list(valid_y_hat)
        print("Logistic regression for calibration...")
        self.platt_reg_ = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.platt_reg_.fit(np.array(probs_cal).reshape(-1, 1), y_cal)
        print("Logistic regression fit done.")
        self.mean_score_ = np.mean(scores)
        self.std_score_ = np.std(scores)
        print(f"Average AUROC: {self.mean_score_}")
        self.model_ = RandomForestClassifier(**hyperparams)
        self.model_.fit(X, y)
        return self

    def _fit_no_pca(self, X, y):
        self.reducer_ = None
        results = self._suggest_best_params(X, y, self.test_size)
        hyperparams = results["best_params"]
        score = results["best_value"]
        hyperparams["n_jobs"] = NUM_CPU
        hyperparams["random_state"] = self.random_state
        hyperparams["class_weight"] = "balanced_subsample"
        print(
            f"Best hyperparameters: {hyperparams}, Inner hyperparameter AUROC: {score}"
        )
        scores = []
        y_cal = []
        probs_cal = []
        for r in range(self.num_splits):
            train_x, valid_x, train_y, valid_y = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state + r,
                stratify=y,
            )
            model_cv = RandomForestClassifier(**hyperparams)
            model_cv.fit(train_x, train_y)
            valid_y_hat = model_cv.predict_proba(valid_x)[:, 1]
            fpr, tpr, _ = roc_curve(valid_y, valid_y_hat)
            auroc = auc(fpr, tpr)
            scores.append(auroc)
            print(f"Internal AUROC CV-{r}: {auroc}")
            y_cal += list(valid_y)
            probs_cal += list(valid_y_hat)
        print("Logistic regression for calibration...")
        self.platt_reg_ = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.platt_reg_.fit(np.array(probs_cal).reshape(-1, 1), y_cal)
        print("Logistic regression fit done.")
        self.mean_score_ = np.mean(scores)
        self.std_score_ = np.std(scores)
        print(f"Average AUROC: {self.mean_score_}")
        self.model_ = RandomForestClassifier(**hyperparams)
        self.model_.fit(X, y)
        return self
    
    def fit(self, X, y):
        if self.pca:
            return self._fit_pca(X, y)
        else:
            return self._fit_no_pca(X, y)

    def predict_proba(self, X):
        if not hasattr(self, "model_") or self.model_ is None:
            raise ValueError("Model not fitted. Call `fit` first.")
        if self.reducer_ is not None:
            X = self.reducer_.transform(X)
        else:
            pass
        y_hat = self.model_.predict_proba(X)[:, 1]
        y_hat = self.platt_reg_.predict_proba(y_hat.reshape(-1, 1))[:, 1]
        return np.vstack((1 - y_hat, y_hat)).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        return roc_auc_score(y, self.predict_proba(X)[:, 1])

    def save_model(self, model_dir: str):
        if not hasattr(self, "model_") or self.model_ is None:
            raise ValueError("Model not fitted. Call `fit` first.")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(self.reducer_, os.path.join(model_dir, "reducer.joblib"))
        joblib.dump(self.model_, model_path)
        joblib.dump(self.platt_reg_, os.path.join(model_dir, "platt_reg.joblib"))

        metadata = {
            "pca": self.pca,
            "random_state": self.random_state,
            "num_splits": self.num_splits,
            "test_size": self.test_size,
            "mean_score_": getattr(self, "mean_score_", None),
            "std_score_": getattr(self, "std_score_", None),
        }
        meta_path = os.path.join(model_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load_model(cls, model_dir: str):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

        model_path = os.path.join(model_dir, "model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        model = joblib.load(model_path)
        platt_reg_path = os.path.join(model_dir, "platt_reg.joblib")
        if not os.path.exists(platt_reg_path):
            raise FileNotFoundError(
                f"Isotonic regression file {platt_reg_path} not found."
            )
        platt_reg = joblib.load(platt_reg_path)
        meta_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file {meta_path} not found.")
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        obj = cls(
            random_state=metadata["random_state"],
            num_splits=metadata["num_splits"],
            test_size=metadata["test_size"],
        )
        obj.model_ = model
        obj.platt_reg_ = platt_reg
        obj.mean_score_ = metadata.get("mean_score_", None)
        obj.std_score_ = metadata.get("std_score_", None)

        return obj


class LazyRandomForestBinaryClassifier(object):
    def __init__(
        self,
        pca: bool = None,
        num_trials: int = 10,
        base_test_size: float = 0.25,
        base_num_splits: int = 3,
        base_timeout: int = 120,
        min_positive_proportion: float = 0.01,
        max_positive_proportion: float = 0.5,
        min_samples: int = 30,
        max_samples: int = 10000,
        min_positive_samples: int = 10,
        max_num_partitions: int = 100,
        min_seen_across_partitions: int = 1,
        force_on_disk: bool = False,
        random_state: int = 42,
    ):
        self.pca = pca
        self.random_state = random_state
        self.base_test_size = base_test_size
        self.base_num_splits = base_num_splits
        self.base_num_trials = num_trials
        self.base_timeout = base_timeout
        self.min_positive_proportion = min_positive_proportion
        self.max_positive_proportion = max_positive_proportion
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.min_positive_samples = min_positive_samples
        self.max_num_partitions = max_num_partitions
        self.min_seen_across_partitions = min_seen_across_partitions
        self.force_on_disk = force_on_disk
        self.fit_time = None
        self.reducers = None
        self.models = None
        self.indices = None

    def _fit_feature_reducer(self, X, y):
        method = self.reducer_method
        max_dim = self.max_reducer_dim
        if X.shape[1] <= max_dim:
            return []
        reducer_0 = VarianceThreshold(threshold=0)
        reducer_0.fit(X)
        X = reducer_0.transform(X)
        if X.shape[1] <= max_dim:
            return [reducer_0]
        if method == "pca":
            max_dim = min(max_dim, X.shape[0])
            reducer_1 = PCA(n_components=max_dim)
            reducer_1.fit(X)
            return [reducer_0, reducer_1]
        elif method == "best":
            reducer_1 = SelectKBest(f_classif, k=max_dim)
            reducer_1.fit(X, y)
            return [reducer_0, reducer_1]
        elif method is None:
            return [reducer_0]
        else:
            raise Exception("Wrong feature reduction method. Use 'pca' or 'best'.")

    def fit(self, X=None, y=None, h5_file=None, h5_idxs=None):
        t0 = time.time()
        iu = InputUtils()
        su = SamplingUtils()
        iu.evaluate_input(
            X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=y, is_y_mandatory=True
        )
        X, h5_file, h5_idxs = iu.preprocessing(
            X=X, h5_file=h5_file, h5_idxs=h5_idxs, force_on_disk=self.force_on_disk
        )
        reducers = []
        models = []
        for idxs in su.get_partition_indices(
            X=X,
            h5_file=h5_file,
            h5_idxs=h5_idxs,
            y=y,
            min_positive_proportion=self.min_positive_proportion,
            max_positive_proportion=self.max_positive_proportion,
            min_samples=self.min_samples,
            max_samples=self.max_samples,
            min_positive_samples=self.min_positive_samples,
            max_num_partitions=self.max_num_partitions,
            min_seen_across_partitions=self.min_seen_across_partitions,
        ):
            if h5_file is not None:
                with h5py.File(h5_file, "r") as f:
                    keys = f.keys()
                    if "values" in keys:
                        values_key = "values"
                    elif "Values" in keys:
                        values_key = "Values"
                    else:
                        raise Exception("HDF5 does not contain a values key")
                    X_sampled = iu.h5_data_reader(
                        f[values_key], [h5_idxs[i] for i in idxs]
                    )
            else:
                X_sampled = X[idxs]
            y_sampled = y[idxs]
            reducer_ = self._fit_feature_reducer(X_sampled, y_sampled)
            for red in reducer_:
                X_sampled = red.transform(X_sampled)
            print(
                f"Fitting model on {len(idxs)} samples, positive samples: {np.sum(y_sampled)}, negative samples: {len(y_sampled) - np.sum(y_sampled)}, number of features {X_sampled.shape[1]}"
            )
            model = BaseRandomForestBinaryClassifier(
                num_splits=self.base_num_splits,
                test_size=self.base_test_size,
                num_trials=self.base_num_trials,
                timeout=self.base_timeout,
                random_state=self.random_state,
            )
            model.fit(X_sampled, y_sampled)
            print("Model fitted.")
            reducers += [reducer_]
            models += [model]
        self.reducers = reducers
        self.models = models
        t1 = time.time()
        self.fit_time = t1 - t0
        print(f"Fitting completed in {self.fit_time:.2f} seconds.")
        return self

    def predict(self, X=None, h5_file=None, h5_idxs=None, chunk_size=1000):
        iu = InputUtils()
        iu.evaluate_input(
            X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=None, is_y_mandatory=False
        )
        X, h5_file, h5_idxs = iu.preprocessing(
            X=X, h5_file=h5_file, h5_idxs=h5_idxs, force_on_disk=self.force_on_disk
        )
        su = SamplingUtils()
        if self.models is None or self.reducers is None:
            raise Exception("No models fitted yet.")
        y_hat = []
        for reducer, model in zip(self.reducers, self.models):
            if h5_file is None:
                n = X.shape[0]
                y_hat_ = []
                for X_chunk in tqdm(
                    su.chunk_matrix(X, chunk_size), desc="Predicting chunks..."
                ):
                    for red in reducer:
                        X_chunk = red.transform(X_chunk)
                    y_hat_ += list(model.predict_proba(X_chunk)[:, 1])
            else:
                n = len(h5_idxs)
                y_hat_ = []
                for X_chunk in tqdm(
                    su.chunk_h5_file(h5_file, h5_idxs, chunk_size),
                    desc="Predicting chunks...",
                ):
                    for red in reducer:
                        X_chunk = red.transform(X_chunk)
                    y_hat_ += list(model.predict_proba(X_chunk)[:, 1])
            y_hat += [y_hat_]
        y_hat = np.array(y_hat).T
        y_hat = np.mean(y_hat, axis=1)
        assert len(y_hat) == n, (
            "Predicted labels length does not match input samples length."
        )
        return y_hat

    def save_model(self, model_dir: str):
        if os.path.exists(model_dir):
            print(f"Model directory already exists: {model_dir}, deleting it...")
            shutil.rmtree(model_dir)
        print(f"Creating model directory: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        if self.models is None:
            raise Exception("No models fitted yet.")
        partition_idx = 0
        for reducer, model in zip(self.reducers, self.models):
            suffix = str(partition_idx).zfill(3)
            partition_dir = os.path.join(model_dir, f"partition_{suffix}")
            os.makedirs(partition_dir, exist_ok=True)
            reducer_path = os.path.join(partition_dir, "reducer.joblib")
            print(f"Saving reducer to {reducer_path}")
            joblib.dump(reducer, reducer_path)
            print(f"Saving model to {partition_dir}")
            model.save_model(partition_dir)
            partition_idx += 1
        metadata = {
            "num_partitions": len(self.models),
            "reducer_method": self.reducer_method,
            "max_reducer_dim": self.max_reducer_dim,
            "random_state": self.random_state,
            "base_test_size": self.base_test_size,
            "base_num_splits": self.base_num_splits,
            "base_num_trials": self.base_num_trials,
            "base_timeout": self.base_timeout,
            "fit_time": self.fit_time,
        }
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def load_model(cls, model_dir: str):
        obj = cls()
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise Exception("Metadata file not found.")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        obj.reducer_method = metadata.get("reducer_method", None)
        obj.max_reducer_dim = metadata.get("max_reducer_dim", None)
        obj.random_state = metadata.get("random_state", None)
        obj.base_test_size = metadata.get("base_test_size", None)
        obj.base_num_splits = metadata.get("base_num_splits", None)
        obj.base_num_trials = metadata.get("base_num_trials", None)
        obj.base_timeout = metadata.get("base_timeout", None)
        obj.fit_time = metadata.get("fit_time", None)
        num_partitions = metadata.get("num_partitions", None)
        if num_partitions <= 0:
            raise Exception("No partitions found in metadata.")
        obj.reducers = []
        obj.models = []
        for i in range(num_partitions):
            suffix = str(i).zfill(3)
            partition_dir = os.path.join(model_dir, f"partition_{suffix}")
            reducer_path = os.path.join(partition_dir, "reducer.joblib")
            print(f"Loading reducer from {reducer_path}")
            reducer = joblib.load(reducer_path)
            print(f"Loading model from {partition_dir}")
            model = BaseRandomForestBinaryClassifier.load_model(partition_dir)
            obj.reducers += [reducer]
            obj.models += [model]
        return obj
