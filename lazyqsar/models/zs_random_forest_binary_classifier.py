from flaml.default import RandomForestClassifier as ZeroShotRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import json
import math
import numpy as np
import optuna
import os
import random
import shutil
import sklearn


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


class BaseZSRFBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        random_state: int = 42,
        num_splits: int = 3,
        test_size: float = 0.25
    ):
        self.random_state = random_state
        self.num_splits = num_splits
        self.test_size = test_size
        self.mean_score_ = None

    def fit(self, X, y):

        scores = []
        for r in range(self.num_splits):
            train_x, valid_x, train_y, valid_y = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state + r, stratify=y
            )
            zero_shot_cv = ZeroShotRandomForestClassifier()
            hyperparams = zero_shot_cv.suggest_hyperparams(train_x, train_y)[0]
            hyperparams['n_jobs'] = 8
            print(f"CV-{r}...: {hyperparams}")
            model_cv = RandomForestClassifier(**hyperparams)
            model_cv.fit(train_x, train_y)
            fpr, tpr, _ = roc_curve(valid_y, model_cv.predict_proba(valid_x)[:, 1])
            auroc = auc(fpr, tpr)
            scores.append(auroc)
            print(f"Internal AUROC zsRF CV-{r}: {auroc}")

        self.mean_score_ = np.mean(scores)
        print(f"Average AUROC: {self.mean_score_}")
        zero_shot_cv = ZeroShotRandomForestClassifier()
        hyperparams = zero_shot_cv.suggest_hyperparams(X, y)[0]
        hyperparams['n_jobs'] = 8
        self.model_ = RandomForestClassifier(**hyperparams)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "model_") or self.model_ is None:
            raise ValueError("Model not fitted. Call `fit` first.")
        return self.model_.predict_proba(X)
    
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

        model_path = os.path.join(model_dir, "rf_model.joblib")
        joblib.dump(self.model_, model_path)

        metadata = {
            "random_state": self.random_state,
            "num_splits": self.num_splits,
            "test_size": self.test_size,
            "mean_score_": getattr(self, "mean_score_", None)
        }
        meta_path = os.path.join(model_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load_model(cls, model_dir: str):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

        # Load the saved RandomForest model
        model_path = os.path.join(model_dir, "rf_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        model = joblib.load(model_path)

        # Load metadata
        meta_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file {meta_path} not found.")
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # Create a new instance and populate it
        obj = cls(
            random_state=metadata["random_state"],
            num_splits=metadata["num_splits"],
            test_size=metadata["test_size"]
        )
        obj.model_ = model
        obj.mean_score_ = metadata.get("mean_score_", None)

        return obj


class LazyZSRandomForestBinaryClassifier(object):

    def __init__(self,
                 reducer_method=None,
                 max_reducer_dim=500,
                 base_test_size=0.25,
                 base_num_splits=3,
                 random_state=42):
        self.reducer_method = reducer_method
        self.max_reducer_dim = max_reducer_dim
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
            model = BaseZSRFBinaryClassifier(num_splits=self.base_num_splits, test_size=self.base_test_size, random_state=self.random_state)
            model.fit(X_sampled, y_sampled)
            print("Model fitted.")
            reducers += [reducer_]
            models += [model]
        self.reducers = reducers
        self.models = models
        self.X_train = X
        self.y_train = y
        return self

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
        obj.random_state = metadata.get("random_state", None)
        
        num_partitions = metadata.get("num_partitions", 0)
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
            model = BaseXGBoostBinaryClassifier.load_model(partition_dir)
            obj.reducers += [reducer]
            obj.models += [model]
        
        return obj