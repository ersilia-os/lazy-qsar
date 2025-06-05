import os
import json
import random
import math
import joblib
import optuna
import sklearn
import shutil
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from scipy.special import expit


class SamplingUtils(object):

    def __init__(self):
        pass

    def chunk_matrix(self, X, chunk_size):
        for i in range(0, X.shape[0], chunk_size):
            yield X[i:i + chunk_size]

    def min_sampling_rounds(self, n_total, n_subsample, target_coverage=0.9):
        n = n_total
        m = n_subsample
        if m <= 0 or n <= 0 or m > n:
            raise ValueError("m must be > 0 and <= n")
        prob_not_seen = 1 - target_coverage
        prob_stay_unseen_per_round = 1 - (m / n)
        if prob_stay_unseen_per_round <= 0:
            return 1
        print(f"Probability of not seeing a sample in one round: {prob_stay_unseen_per_round}")
        rounds = math.log(prob_not_seen) / math.log(prob_stay_unseen_per_round)
        return math.ceil(rounds)

    def get_partition_indices(self, y,
                            min_positive_proportion=0.01,
                            max_positive_proportion=0.5,
                            min_samples=30,
                            max_samples=10000,
                            min_positive_samples=10,
                            max_num_partitions=100):
        pos_idxs = [i for i, y_ in enumerate(y) if y_ == 1]
        neg_idxs = [i for i, y_ in enumerate(y) if y_ == 0]
        random.shuffle(pos_idxs)
        random.shuffle(neg_idxs)
        n_tot = len(y)
        n_pos = len(pos_idxs)
        n_neg = len(neg_idxs)
        print(f"Total samples: {n_tot}, positive samples: {n_pos}, negative samples: {n_neg}")
        print(f"Positive proportion: {n_pos / n_tot:.2f}")
        if n_tot < min_samples:
            raise Exception("Not enough samples.")
        n_samples = min(n_tot, max_samples)
        real_pos_prop = n_pos / n_tot
        upper_pos_prop = min(n_pos / n_samples, max_positive_proportion)
        if real_pos_prop <= min_positive_proportion:
            if upper_pos_prop <= min_positive_proportion:
                desired_pos_prop = min_positive_proportion
            else:
                desired_pos_prop = upper_pos_prop
        elif real_pos_prop >= max_positive_proportion:
            desired_pos_prop = max_positive_proportion
        else:
            desired_pos_prop = upper_pos_prop
        n_pos_samples = min(int(n_samples * desired_pos_prop) + 1, n_pos)
        n_neg_samples = min(n_neg, int(n_pos_samples * (1 - desired_pos_prop) / desired_pos_prop) + 1, n_samples - n_pos_samples)
        n_samples = n_pos_samples + n_neg_samples
        if n_pos_samples < min_positive_samples:
            raise Exception("Not enough positive samples.")
        if n_neg_samples < min_positive_samples:
            raise Exception("Not enough negative samples.")
        pos_sampling_rounds = min(self.min_sampling_rounds(n_pos, n_pos_samples), max_num_partitions)
        neg_sampling_rounds = min(self.min_sampling_rounds(n_neg, n_neg_samples), max_num_partitions)
        sampling_rounds = max(pos_sampling_rounds, neg_sampling_rounds)
        print(f"Sampling rounds: {sampling_rounds}, positive samples per round: {n_pos_samples}, negative samples per round: {n_neg_samples}")
        print(f"Desired positive proportion: {desired_pos_prop}", "Actual positive proportion: ", n_pos_samples / n_samples)
        for _ in range(sampling_rounds):
            pos_idxs_sampled = random.sample(pos_idxs, n_pos_samples)
            neg_idxs_sampled = random.sample(neg_idxs, n_neg_samples)
            sampled_idxs = pos_idxs_sampled + neg_idxs_sampled
            random.shuffle(sampled_idxs)
            yield sampled_idxs


class BaseXGBoostBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_trials: int = 1000,
        timeout: int = 600,
        random_state: int = 42,
        num_splits: int = 3,
        test_size: float = 0.25
    ):
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.num_splits = num_splits
        self.test_size = test_size
        self.best_params_ = None
        self.best_score_ = None
        self._booster = None

    def _objective(self, trial, X, y):

        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "tree_method": "exact",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] in ["gbtree", "dart"]:
            param.update({
                "max_depth": trial.suggest_int("max_depth", 3, 9, step=2),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            })

        if param["booster"] == "dart":
            param.update({
                "sample_type": trial.suggest_categorical("sample_type", ["uniform", "weighted"]),
                "normalize_type": trial.suggest_categorical("normalize_type", ["tree", "forest"]),
                "rate_drop": trial.suggest_float("rate_drop", 1e-8, 1.0, log=True),
                "skip_drop": trial.suggest_float("skip_drop", 1e-8, 1.0, log=True),
            })

        scores = []
        for _ in range(self.num_splits):
            train_x, valid_x, train_y, valid_y = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            dtrain = xgb.DMatrix(train_x, label=train_y)
            dvalid = xgb.DMatrix(valid_x, label=valid_y)

            bst = xgb.train(param, dtrain)
            preds = bst.predict(dvalid)
            score = sklearn.metrics.roc_auc_score(valid_y, preds)
            scores += [float(score)]
        return np.mean(scores)

    def fit(self, X, y):
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction="maximize",
            study_name=None,
            sampler=sampler,
        )

        print("Fitting...")
        best_score = -np.inf
        trials_without_improvement = 0
        improvement_threshold = 0.01
        patience = 500
        early_stopping = False
        baseline_score_for_patience = best_score

        def objective_with_custom_early_stop(trial):
            nonlocal best_score, trials_without_improvement, early_stopping, baseline_score_for_patience
            if early_stopping:
                # print("Skipping trial due to early stopping criteria.")
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
                # print(f"Early stopping: No significant improvement in the last {patience} trials.")
                raise optuna.exceptions.TrialPruned()
            return score
        
        study.optimize(
            objective_with_custom_early_stop,
            n_trials=self.n_trials,
            timeout=self.timeout
        )

        self.best_params_ = study.best_params
        self.best_score_ = study.best_value

        print(f"Best AUROC: {self.best_score_:.4f}")
        dtrain = xgb.DMatrix(X, label=y)
        self._booster = xgb.train(self.best_params_, dtrain)
        return self

    def predict_proba(self, X):
        if self._booster is None:
            raise ValueError("Model not fitted, call fit() first.")
        dmat = xgb.DMatrix(X)
        raw_score = self._booster.predict(dmat)
        proba = expit(raw_score)
        return np.array(np.vstack([1 - proba, proba]).T)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        preds = self.predict_proba(X)[:, 1]
        return float(sklearn.metrics.roc_auc_score(y, preds))

    def save_model(self, model_dir: str):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self._booster.save_model(os.path.join(model_dir, "booster.json"))
        init_data = {
            "n_trials": self.n_trials,
            "timeout": self.timeout,
            "random_state": self.random_state,
            "num_splits": self.num_splits,
            "test_size": self.test_size,
        }
        meta = {
            "best_params_": self.best_params_,
            "best_score_": self.best_score_,
            "init_data_": init_data,
        }
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def load_model(cls, model_dir: str):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            meta = json.load(f)

        booster = xgb.Booster()
        booster.load_model(os.path.join(model_dir, "booster.json"))

        obj = cls()
        obj.best_params_ = meta["best_params_"]
        obj.best_score_ = meta["best_score_"]
        obj._booster = booster
        obj.n_trials = meta["init_data_"]["n_trials"]
        obj.timeout = meta["init_data_"]["timeout"]
        obj.random_state = meta["init_data_"]["random_state"]
        obj.num_splits = meta["init_data_"]["num_splits"]
        obj.test_size = meta["init_data_"]["test_size"]

        return obj


class LazyXGBoostBinaryClassifier(object):

    def __init__(self,
                 reducer_method=None,
                 max_reducer_dim=500,
                 base_num_trials=1000,
                 base_timeout=600,
                 base_test_size=0.25,
                 base_num_splits=3,
                 random_state=42):
        self.reducer_method = reducer_method
        self.max_reducer_dim = max_reducer_dim
        self.base_n_trials = base_num_trials
        self.base_timeout = base_timeout
        self.random_state = random_state
        self.base_test_size = base_test_size
        self.base_num_splits = base_num_splits

        self.reducers = None
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

    def fit(self, X, y):
        su = SamplingUtils()
        reducers = []
        models = []
        for idxs in su.get_partition_indices(y):
            X_sampled = X[idxs]
            y_sampled = y[idxs]
            reducer_ = self._fit_feature_reducer(X_sampled, y_sampled)
            for red in reducer_:
                X_sampled = red.transform(X_sampled)
            print(f"Fitting model on {len(idxs)} samples, positive samples: {np.sum(y_sampled)}, negative samples: {len(y_sampled) - np.sum(y_sampled)}, number of features {X_sampled.shape[1]}")
            model = BaseXGBoostBinaryClassifier(n_trials=self.base_n_trials, timeout=self.base_timeout, num_splits=self.base_num_splits, test_size=self.base_test_size, random_state=self.random_state)
            model.fit(X_sampled, y_sampled)
            print("Model fitted.")
            reducers += [reducer_]
            models += [model]
        self.reducers = reducers
        self.models = models
        self.X_train = X
        self.y_train = y

    def predict(self, X, chunk_size=1000):
        su = SamplingUtils()
        if self.models is None or self.reducers is None:
            raise Exception("No models fitted yet.")
        y_hat = []
        for reducer, model in zip(self.reducers, self.models):
            y_hat_ = []
            for X_chunk in su.chunk_matrix(X, chunk_size):
                print("Predicting chunk of size:", X_chunk.shape[0])
                for red in reducer:
                    X_chunk = red.transform(X_chunk)
                y_hat_ += list(model.predict_proba(X_chunk)[:,1])
            y_hat += [y_hat_]
        y_hat = np.array(y_hat).T
        y_hat = np.mean(y_hat, axis=1)
        assert len(y_hat) == X.shape[0], "Predicted labels length does not match input samples length."
        return y_hat

    def save_model(self, model_dir: str):
        if not os.path.exists(model_dir):
            print(f"Creating model directory: {model_dir}")
            os.makedirs(model_dir)
        else:
            print(f"Model directory already exists: {model_dir}")
            shutil.rmtree(model_dir)
        if self.models is None:
            raise Exception("No models fitted yet.")
        partition_idx = 0
        for reducer, model in zip(self.reducers, self.models):
            sufix = str(partition_idx).zfill(3)
            partition_dir = os.path.join(model_dir, f"partition_{sufix}")
            if not os.path.exists(partition_dir):
                print(f"Creating partition directory: {partition_dir}")
                os.makedirs(partition_dir)
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
            "base_n_trials": self.base_n_trials,
            "base_timeout": self.base_timeout,
            "random_state": self.random_state,
        }
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    @classmethod
    def load_model(cls, model_dir: str):
        
        obj = cls()

        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise Exception("Metadata file not found.")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        obj.reducer_method = metadata.get("reducer_method", None)
        obj.max_reducer_dim = metadata.get("max_reducer_dim", None)
        obj.base_n_trials = metadata.get("base_n_trials", None)
        obj.base_timeout = metadata.get("base_timeout", None)
        obj.random_state = metadata.get("random_state", None)
        
        num_partitions = metadata.get("num_partitions", 0)
        if num_partitions <= 0:
            raise Exception("No partitions found in metadata.")
        obj.reducers = []
        obj.models = []
        for i in range(num_partitions):
            sufix = str(i).zfill(3)
            partition_dir = os.path.join(model_dir, f"partition_{sufix}")
            reducer_path = os.path.join(partition_dir, "reducer.joblib")
            print(f"Loading reducer from {reducer_path}")
            reducer = joblib.load(reducer_path)
            print(f"Loading model from {partition_dir}")
            model = BaseXGBoostBinaryClassifier.load_model(partition_dir)
            obj.reducers += [reducer]
            obj.models += [model]
        
        return obj