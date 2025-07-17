import joblib
import json
import time
import numpy as np
import os
import shutil
import h5py
import multiprocessing
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from .utils import BinaryClassifierSamplingUtils as SamplingUtils
from .utils import InputUtils
from .utils import StratifiedKFolder

import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


NUM_CPU = max(1, multiprocessing.cpu_count() - 1)


class BaseLogisticRegressionBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        pca: bool = False,
        num_splits: int = 3,
        test_size: float = 0.25,
        random_state: int = 42,
        num_trials: int = 50,
        timeout: int = 600,
        max_positive_proportion: float = 0.5
    ):
        self.pca = pca
        self.random_state = random_state
        self.num_splits = num_splits
        self.test_size = test_size
        self.num_trials = num_trials
        self.timeout = timeout
        self.max_positive_proportion = max_positive_proportion
        self.mean_score_ = None

    def _objective(self, trial, X, y):

        n_components = trial.suggest_float("n_components", 0.5, 0.99, step=0.01)

        num_splits = max(self.num_splits, int(1 / self.test_size))

        cv = StratifiedKFolder(n_splits=num_splits, max_positive_proportion=self.max_positive_proportion, random_state=self.random_state)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("clf", LogisticRegressionCV(class_weight="balanced", 
                                         refit=False, n_jobs=NUM_CPU,
                                         cv=cv, random_state=self.random_state,
                                         scoring="roc_auc"))
        ])

        scores = []
        for _ in range(3):
            scores += [cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=NUM_CPU).mean()]

        return np.mean(scores)
    
    def _suggest_best_params(self, X, y, test_size):

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction="maximize",
            study_name=None,
            sampler=sampler,
        )

        n_components = 0.999
        hyperparams = {
            "n_components": n_components,
        }

        study.enqueue_trial(hyperparams)

        print("Fitting...")
        best_score = -np.inf
        trials_without_improvement = 0
        improvement_threshold = 0.01
        patience = max(5, self.num_trials // 5)
        early_stopping = False
        baseline_score_for_patience = best_score

        def objective_with_custom_early_stop(trial):
            nonlocal best_score, trials_without_improvement, early_stopping, baseline_score_for_patience
            if early_stopping:
                print("Skipping trial due to early stopping criteria.")
                raise optuna.exceptions.TrialPruned()
            score = self._objective(trial, X, y)
            if score > best_score:
                best_score = score
            if score > baseline_score_for_patience + improvement_threshold:
                trials_without_improvement = 0
                baseline_score_for_patience = score
            else:
                trials_without_improvement += 1
            if trials_without_improvement >= patience:
                early_stopping = True
                print(f"Early stopping: No significant improvement in the last {patience} trials.")
                raise optuna.exceptions.TrialPruned()
            return score
        
        study.optimize(
            objective_with_custom_early_stop,
            n_trials=self.num_trials,
            timeout=self.timeout
        )

        results = {
            "best_params": study.best_params,
            "best_value": study.best_value,
        }
        
        return results

    def _fit_pca(self, X, y):
        results = self._suggest_best_params(X, y, self.test_size)
        n_components = results['best_params']['n_components']
        hyperparams = results['best_params']
        score = results['best_value']
        hyperparams['n_jobs'] = NUM_CPU
        hyperparams['random_state'] = self.random_state
        hyperparams['class_weight'] = 'balanced'
        print(f"Best hyperparameters: {hyperparams}, Inner hyperparameter AUROC: {score}")

        print("Fitting on data with shape:", X.shape)
        scores = []
        num_splits = self.num_splits
        test_size = self.test_size
        if num_splits is None:
            num_splits = 3
        if test_size is None:
            test_size = 0.25
        num_splits = max(num_splits, int(1 / test_size))
        cv = StratifiedKFolder(n_splits=num_splits, shuffle=True, random_state=self.random_state, max_positive_proportion=self.max_positive_proportion)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("clf", LogisticRegressionCV(class_weight="balanced",
                                         refit=True, n_jobs=NUM_CPU,
                                         cv=cv, random_state=self.random_state,
                                         scoring="roc_auc"))
        ])
    
        pipe.fit(X, y)
        model_cv = pipe.named_steps["clf"]
        if hasattr(model_cv, "scores_"):
            scores_dict = model_cv.scores_
            if 1 in scores_dict:
                scores = scores_dict[1].mean(axis=1)
            else:
                scores = list(scores_dict.values())[0].mean(axis=1)
        else:
            scores = []
        print(f"Logistic regression fit done.")
        self.mean_score_ = np.mean(scores)
        self.std_score_ = np.std(scores)
        print(f"Average AUROC: {self.mean_score_}")

        best_C = model_cv.C_[0]
        solver = model_cv.solver
        penalty = model_cv.penalty
        fit_intercept = model_cv.fit_intercept
        class_weight = model_cv.class_weight
        intercept_scaling = getattr(model_cv, 'intercept_scaling', 1)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("clf", LogisticRegression(
                C=best_C,
                solver=solver,
                penalty=penalty,
                fit_intercept=fit_intercept,
                class_weight=class_weight,
                intercept_scaling=intercept_scaling,
                max_iter=1000
            ))
        ])
        pipe.fit(X, y)
        self.model_ = pipe
        return self
    
    def _fit_no_pca(self, X, y):
        scores = []
        num_splits = self.num_splits
        test_size = self.test_size
        if num_splits is None:
            num_splits = 3
        if test_size is None:
            test_size = 0.25
        num_splits = max(num_splits, int(1 / test_size))
        cv = cv = StratifiedKFolder(n_splits=num_splits, shuffle=True, random_state=self.random_state, max_positive_proportion=self.max_positive_proportion)
        model_cv = LogisticRegressionCV(
            class_weight="balanced",
            refit=True,
            n_jobs=NUM_CPU,
            cv=cv,
            random_state=self.random_state,
            scoring="roc_auc"
        )
        model_cv.fit(X, y)
        if hasattr(model_cv, "scores_"):
            scores_dict = model_cv.scores_
            if 1 in scores_dict:
                scores = scores_dict[1].mean(axis=1)
            else:
                scores = list(scores_dict.values())[0].mean(axis=1)
        else:
            scores = []
        print(f"Logistic regression fit done.")
        self.mean_score_ = np.mean(scores)
        self.std_score_ = np.std(scores)
        print(f"Average AUROC: {self.mean_score_}")

        best_C = model_cv.C_[0]
        solver = model_cv.solver
        penalty = model_cv.penalty
        fit_intercept = model_cv.fit_intercept
        class_weight = model_cv.class_weight
        intercept_scaling = getattr(model_cv, 'intercept_scaling', 1)
        model = LogisticRegression(
                    C=best_C,
                    solver=solver,
                    penalty=penalty,
                    fit_intercept=fit_intercept,
                    class_weight=class_weight,
                    intercept_scaling=intercept_scaling,
                    max_iter=1000
                )
        model.fit(X, y)
        self.model_ = model
        return self

    def fit(self, X, y):
        if self.pca:
            return self._fit_pca(X, y)
        else:
            return self._fit_no_pca(X, y)

    def predict_proba(self, X):
        if not hasattr(self, "model_") or self.model_ is None:
            raise ValueError("Model not fitted. Call `fit` first.")
        y_hat = self.model_.predict_proba(X)[:,1]
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
        joblib.dump(self.model_, model_path)

        metadata = {
            "random_state": self.random_state,
            "num_splits": self.num_splits,
            "test_size": self.test_size,
            "num_trials": self.num_trials,
            "mean_score_": getattr(self, "mean_score_", None),
            "std_score_": getattr(self, "std_score_", None)
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
        meta_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file {meta_path} not found.")
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        obj = cls(
            random_state=metadata["random_state"],
            num_splits=metadata["num_splits"],
            test_size=metadata["test_size"],
            num_trials=metadata["num_trials"]
        )
        obj.model_ = model
        obj.mean_score_ = metadata.get("mean_score_", None)
        obj.std_score_ = metadata.get("std_score_", None)

        return obj


class LazyLogisticRegressionBinaryClassifier(object):

    def __init__(self,
                 pca: bool = False,
                 num_trials: int = 10,
                 base_test_size: float = 0.25,
                 base_num_splits: int = 3,
                 min_positive_proportion: float = 0.01,
                 max_positive_proportion: float = 0.5,
                 min_samples: int = 30,
                 max_samples: int = 10000,
                 min_positive_samples: int = 10,
                 max_num_partitions: int = 100,
                 min_seen_across_partitions: int = 1,
                 force_max_positive_proportion_at_partition: bool = False,
                 force_on_disk: bool = False,
                 random_state: int = 42):
        self.pca = pca
        self.random_state = random_state
        self.base_test_size = base_test_size
        self.base_num_splits = base_num_splits
        self.base_num_trials = num_trials
        self.min_positive_proportion = min_positive_proportion
        self.max_positive_proportion = max_positive_proportion
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.min_positive_samples = min_positive_samples
        self.max_num_partitions = max_num_partitions
        self.min_seen_across_partitions = min_seen_across_partitions
        self.force_max_positive_proportion_at_partition = force_max_positive_proportion_at_partition
        self.force_on_disk = force_on_disk
        self.fit_time = None
        self.reducers = None
        self.models = None
        self.indices = None

    def _fit_feature_reducer(self, X, y):
        reducer_0 = VarianceThreshold(threshold=0)
        reducer_0.fit(X)
        return [reducer_0]

    def fit(self, X=None, y=None, h5_file=None, h5_idxs=None):
        t0 = time.time()
        iu = InputUtils()
        su = SamplingUtils()
        iu.evaluate_input(X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=y, is_y_mandatory=True)
        X, h5_file, h5_idxs = iu.preprocessing(X=X, h5_file=h5_file, h5_idxs=h5_idxs, force_on_disk=self.force_on_disk)
        reducers = []
        models = []
        for idxs in su.get_partition_indices(X=X,
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
                                             force_max_positive_proportion_at_partition=self.force_max_positive_proportion_at_partition):
            if h5_file is not None:
                with h5py.File(h5_file, "r") as f:
                    X_sampled = iu.h5_data_reader(f["values"], [h5_idxs[i] for i in idxs])
            else:
                X_sampled = X[idxs]
            y_sampled = y[idxs]
            reducer_ = self._fit_feature_reducer(X_sampled, y_sampled)
            for red in reducer_:
                X_sampled = red.transform(X_sampled)
            print(f"Fitting model on {len(idxs)} samples, positive samples: {np.sum(y_sampled)}, negative samples: {len(y_sampled) - np.sum(y_sampled)}, number of features {X_sampled.shape[1]}")
            model = BaseLogisticRegressionBinaryClassifier(pca=self.pca, num_splits=self.base_num_splits, test_size=self.base_test_size, num_trials=self.base_num_trials, random_state=self.random_state, max_positive_proportion=self.max_positive_proportion)
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
        iu.evaluate_input(X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=None, is_y_mandatory=False)
        X, h5_file, h5_idxs = iu.preprocessing(X=X, h5_file=h5_file, h5_idxs=h5_idxs, force_on_disk=self.force_on_disk)
        su = SamplingUtils()
        if self.models is None or self.reducers is None:
            raise Exception("No models fitted yet.")
        y_hat = []
        for reducer, model in zip(self.reducers, self.models):
            if h5_file is None:
                n = X.shape[0]
                y_hat_ = []
                for X_chunk in tqdm(su.chunk_matrix(X, chunk_size), desc="Predicting chunks..."):
                    for red in reducer:
                        X_chunk = red.transform(X_chunk)
                    y_hat_ += list(model.predict_proba(X_chunk)[:,1])
            else:
                n = len(h5_idxs)
                y_hat_ = []
                for X_chunk in tqdm(su.chunk_h5_file(h5_file, h5_idxs, chunk_size), desc="Predicting chunks..."):
                    for red in reducer:
                        X_chunk = red.transform(X_chunk)
                    y_hat_ += list(model.predict_proba(X_chunk)[:,1])        
            y_hat += [y_hat_]
        y_hat = np.array(y_hat).T
        y_hat = np.mean(y_hat, axis=1)
        assert len(y_hat) == n, "Predicted labels length does not match input samples length."
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
            "pca": self.pca,
            "random_state": self.random_state,
            "base_test_size": self.base_test_size,
            "base_num_splits": self.base_num_splits,
            "base_num_trials": self.base_num_trials,
            "fit_time": self.fit_time
        }
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def load_model(cls, model_dir: str):
        obj = cls()
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise Exception("Metadata file not found.")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        obj.pca = metadata.get("pca", None)
        obj.random_state = metadata.get("random_state", None)
        obj.base_test_size = metadata.get("base_test_size", None)
        obj.base_num_splits = metadata.get("base_num_splits", None)
        obj.base_num_trials = metadata.get("base_num_trials", None)
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
            model = BaseLogisticRegressionBinaryClassifier.load_model(partition_dir)
            obj.reducers += [reducer]
            obj.models += [model]
        return obj