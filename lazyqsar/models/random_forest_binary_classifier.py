import joblib
import json
import math
import numpy as np
import optuna
import os
import random
import shutil
import sklearn
import h5py
import psutil
import multiprocessing
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.model_selection import StratifiedKFold
from flaml.default import RandomForestClassifier as ZeroShotRandomForestClassifier

NUM_CPU = max(1, multiprocessing.cpu_count() - 1)


class InputUtils(object):

    def __init__(self):
        pass

    def evaluate_input(self, X=None, h5_file=None, h5_idxs=None, y=None, is_y_mandatory=True):
        if is_y_mandatory:
            if y is None:
                raise ValueError("y cannot be None. Provide a label vector.")
        if X is None and h5_file is None:
            raise ValueError("Either X or h5_file must be provided.")
        if X is not None and h5_file is not None:
            raise ValueError("Provide either X or h5_file, not both.")
        if h5_file is not None:
            if not os.path.exists(h5_file):
                raise FileNotFoundError(f"File {h5_file} does not exist.")
            if not h5_file.endswith(".h5"):
                raise ValueError("h5_file should be a .h5 file.")
            if h5_idxs is None:
                with h5py.File(h5_file, "r") as f:
                    if "values" not in f:
                        raise ValueError("h5_file must contain 'values' dataset.")
                    h5_idxs = [i for i in range(f["values"].shape[0])]
            else:
                if y is not None:
                    if len(h5_idxs) != len(y):
                        raise Exception("h5_idxs length must match y length.")
        if X is not None and h5_idxs is not None:
            raise Exception("You cannot provide h5_idxs if X is provided. Use X only or h5_file with h5_idxs.")

    def h5_data_reader(self, x_data, idxs):
        sorted_indices = np.argsort(idxs)
        sorted_idxs = np.array(idxs)[sorted_indices]
        sorted_data = x_data[sorted_idxs, :]
        inverse_sort = np.argsort(sorted_indices)
        x = sorted_data[inverse_sort]
        return x
    
    def is_load_full_h5_file(self, h5_file):
        with h5py.File(h5_file, 'r') as f:
            dataset = f["values"]
            if isinstance(dataset, h5py.Dataset):
                size_bytes = dataset.size * dataset.dtype.itemsize
                size_gb = size_bytes / (1024 ** 3)
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        print(f"Available memory: {available_gb:.2f} GB, H5 file size: {size_gb:.2f} GB")
        if available_gb > size_gb * 1.5:
            return True
        else:
            return False
        
    def preprocessing(self, X=None, h5_file=None, h5_idxs=None):
        if h5_file is not None:
            if h5_idxs is None:
                with h5py.File(h5_file, "r") as f:
                    if "values" not in f:
                        raise ValueError("h5_file must contain 'values' dataset.")
                    h5_idxs = [i for i in range(f["values"].shape[0])]
            if self.is_load_full_h5_file(h5_file):
                print("Loading full h5 file into memory...")
                with h5py.File(h5_file, "r") as f:
                    X = f["values"][:]
                    X = X[h5_idxs, :]
                    h5_file = None
                    h5_idxs = None
        return X, h5_file, h5_idxs


class SamplingUtils(object):

    def __init__(self):
        pass

    def chunk_h5_file(self, h5_file, h5_idxs, chunk_size):
        iu = InputUtils()
        with h5py.File(h5_file, "r") as f:
            if "values" not in f:
                raise ValueError("h5_file must contain 'values' dataset.")
            values = f["values"]
            for i in range(0, len(h5_idxs), chunk_size):
                idxs_chunk = h5_idxs[i:i + chunk_size]
                yield iu.h5_data_reader(values, idxs_chunk)

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
    
    @staticmethod
    def is_integer_matrix(X):
        X_ = X[:10]
        if np.issubdtype(X_.dtype, np.integer):
            return True
        if np.issubdtype(X_.dtype, int):
            return True
        return np.all(np.equal(X_, np.floor(X_)))

    def quick_auc_estimate(self, X, y):
        if self.is_integer_matrix(X):
            model = BernoulliNB()
        else:
            model = GaussianNB()
        X = np.array(X)
        y = np.array(y)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]
            scores.append(roc_auc_score(y_test, preds))
        return float(np.mean(scores))
    
    @staticmethod
    def _suggest_row(idxs_matrix, accepted_rows, sampled_idxs_counts):
        accepted_rows_set = set(accepted_rows)
        min_count = np.min([v for _, v in sampled_idxs_counts.items()])
        if min_count >= 3:
            return None
        must_sample_idxs = []
        for k,v in sampled_idxs_counts.items():
            if v == min_count:
                must_sample_idxs += [k]
        must_sample_idxs = set(must_sample_idxs)
        for i in range(idxs_matrix.shape[0]):
            if i in accepted_rows_set:
                continue
            row = idxs_matrix[i, :]
            for v in row:
                if v in must_sample_idxs:
                    return i
    
    def get_partition_indices(self,
                              X=None,
                              h5_file=None,
                              h5_idxs=None,
                              y=None,
                              min_positive_proportion=0.01,
                              max_positive_proportion=0.5,
                              min_samples=30,
                              max_samples=10000,
                              min_positive_samples=10,
                              max_num_partitions=100):
        
        iu = InputUtils()
        iu.evaluate_input(X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=y, is_y_mandatory=True)
        
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
        effective_sampling_rounds = sampling_rounds*3
        print(f"Effective sampling rounds: {effective_sampling_rounds}")
        idxs_matrix = np.zeros((effective_sampling_rounds, n_samples), dtype=int)
        for i in range(effective_sampling_rounds):
            pos_idxs_sampled = random.sample(pos_idxs, n_pos_samples)
            neg_idxs_sampled = random.sample(neg_idxs, n_neg_samples)
            sampled_idxs = pos_idxs_sampled + neg_idxs_sampled
            idxs_matrix[i, :] = sorted(sampled_idxs)
        idxs_matrix = np.unique(idxs_matrix, axis=0)
        print(f"Unique sampled indices matrix shape: {idxs_matrix.shape}")
        if idxs_matrix.shape[0] > sampling_rounds:
            if h5_file:
                with h5py.File(h5_file, "r") as f:
                    auc_estimates = []
                    for i in tqdm(range(idxs_matrix.shape[0])):
                        idxs_y = idxs_matrix[i, :]
                        idxs_x = [h5_idxs[idx] for idx in idxs_y]
                        X_in = iu.h5_data_reader(f["values"], idxs_x)
                        y_in = [y[idx] for idx in idxs_y]
                        auc_est = self.quick_auc_estimate(X_in, y_in)
                        auc_estimates += [auc_est]
            else:
                auc_estimates = []
                for i in tqdm(range(idxs_matrix.shape[0])):
                    idxs_y = idxs_matrix[i, :]
                    X_in = X[idxs_y, :]
                    y_in = [y[idx] for idx in idxs_y]
                    auc_est = self.quick_auc_estimate(X_in, y_in)
                    auc_estimates += [auc_est]
            print(f"Estimated AUCs for {len(auc_estimates)} partitions: {auc_estimates[:10]}")
            sorted_indices = np.argsort(auc_estimates)[::-1]
            auc_estimates = [auc_estimates[i] for i in sorted_indices]
            idxs_matrix = idxs_matrix[sorted_indices]
            all_sampled_idxs = list(set([int(x) for x in list(idxs_matrix.ravel())]))
            sampled_idxs_counts = {idx: 0 for idx in all_sampled_idxs}
            accepted_rows = []
            for i in range(sampling_rounds):
                row_idx = self._suggest_row(idxs_matrix, accepted_rows, sampled_idxs_counts)
                if row_idx is None:
                    print("No need to sample more rows")
                    continue
                for v in idxs_matrix[row_idx, :]:
                    sampled_idxs_counts[v] += 1
                accepted_rows += [row_idx]
            print(f"Accepted rows: {len(accepted_rows)} out of {idxs_matrix.shape[0]} total rows.")
            print(f"Accepted rows: {accepted_rows}")
            print(f"AUC estimates of accepted rows: {[auc_estimates[i] for i in accepted_rows][:10]}")
            assert len(accepted_rows) <= sampling_rounds, "Too many accepted rows."
            accepted_rows = [i for i in accepted_rows]
            random.shuffle(accepted_rows)
            idxs_matrix_ = np.zeros((len(accepted_rows), idxs_matrix.shape[1]), dtype=int)
            for i, row_idx in enumerate(accepted_rows):
                idxs_matrix_[i, :] = idxs_matrix[row_idx, :]
        else:
            idxs_matrix_ = np.zeros((idxs_matrix.shape[0], idxs_matrix.shape[1]), dtype=int)
            for i in range(idxs_matrix.shape[0]):
                r = idxs_matrix[i, :]
                random.shuffle(r)
                idxs_matrix_[i, :] = r
        print(f"Final indices matrix shape: {idxs_matrix_.shape}")
        for i in range(idxs_matrix_.shape[0]):
            idxs = [int(x) for x in idxs_matrix_[i, :]]
            yield idxs


class BaseRandomForestBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        num_splits: int = 3,
        test_size: float = 0.25,
        num_trials: int = 100,
        timeout: int = 600,
        random_state: int=42
    ):
        self.random_state = random_state
        self.num_splits = num_splits
        self.test_size = test_size
        self.num_trials = num_trials
        self.timeout = timeout
        self.mean_score_ = None

    def _suggest_param_search(self, hyperparams, n_samples, n_features, test_size):
        # suggesting range of n_estimators
        def_n_estimators = hyperparams["n_estimators"]
        min_n_estimators = max(10, int(def_n_estimators) - 100)
        max_n_estimators = min(1000, int(def_n_estimators) + 100)
        # suggesting range of max_features
        def_max_features = hyperparams["max_features"]
        if def_max_features is None:
            def_max_features = 1.0
        elif str(def_max_features) == "sqrt":
            def_max_features = np.sqrt(n_features) / n_features
        elif str(def_max_features) == "log2":
            def_max_features = np.log2(n_features) / n_features
        elif str(def_max_features) == "auto":
            def_max_features = np.sqrt(n_features) / n_features # if auto, assume sqrt
        elif def_max_features > 1:
            def_max_features = def_max_features / n_features # if it is above 1, assume it is an absolute number
        else:
            pass
        min_max_features = max(0.05, def_max_features - 0.2)
        max_max_features = min(1.0, def_max_features + 0.2)
        # suggesting max leaf nodes
        def_max_leaf_nodes = hyperparams["max_leaf_nodes"]
        def_max_leaf_nodes_log2 = math.log2(def_max_leaf_nodes) if def_max_leaf_nodes > 0 else 0
        min_max_leaf_nodes_log2 = max(4, def_max_leaf_nodes_log2 - 1)
        max_max_leaf_nodes_log2 = min(10, def_max_leaf_nodes_log2 + 1)
        min_max_leaf_nodes = int(np.round(2 ** min_max_leaf_nodes_log2, 0))
        max_max_leaf_nodes = int(np.round(2 ** max_max_leaf_nodes_log2, 0))
        min_max_leaf_nodes = min(min_max_leaf_nodes, int(n_samples*(1-test_size)))
        max_max_leaf_nodes = max(max_max_leaf_nodes, max(int(n_samples*(1-test_size)), def_max_leaf_nodes))
        # criterion
        if "criterion" in hyperparams:
            criterion = hyperparams["criterion"]
        else:
            criterion = "gini"
        # sanity check (should not happen)
        if min_n_estimators == max_n_estimators:
            max_n_estimators = min_n_estimators + 1
        if min_max_features == max_max_features:
            min_max_features = min_max_features-0.01
        if min_max_leaf_nodes == max_max_leaf_nodes:
            min_max_leaf_nodes = min_max_leaf_nodes - 1
        # preparing the parameters
        param = {
            "n_estimators": sorted([min_n_estimators, max_n_estimators]),
            "max_features": sorted([min_max_features, max_max_features]), 
            "max_leaf_nodes": sorted([min_max_leaf_nodes, max_max_leaf_nodes]),
            "criterion": [criterion]
        }
        return param

    def _objective(self, trial, X, y, param):
        param = {"n_estimators": trial.suggest_int("n_estimators", param["n_estimators"][0], param["n_estimators"][1]),
                 "max_features": trial.suggest_float("max_features", param["max_features"][0], param["max_features"][1]),
                 "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", param["max_leaf_nodes"][0], param["max_leaf_nodes"][1], log=True),
                 "criterion": trial.suggest_categorical("criterion", param["criterion"])}
        param["n_jobs"] = NUM_CPU
        scores = []
        for _ in range(self.num_splits):
            train_x, valid_x, train_y, valid_y = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
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
        print("Suggested zero-shot hyperparameters:", hyperparams)
        n_samples, n_features = X.shape
        hyperparam_search = self._suggest_param_search(hyperparams, n_samples, n_features, test_size)

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
            nonlocal best_score, trials_without_improvement, early_stopping, baseline_score_for_patience
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

    def fit(self, X, y):
        results = self._suggest_best_params(X, y, self.test_size)
        hyperparams = results['best_params']
        score = results['best_value']
        hyperparams['n_jobs'] = NUM_CPU
        hyperparams['random_state'] = self.random_state
        print(f"Best hyperparameters: {hyperparams}, Inner hyperparameter AUROC: {score}")
        scores = []
        for r in range(self.num_splits):
            train_x, valid_x, train_y, valid_y = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state + r, stratify=y
            )
            model_cv = RandomForestClassifier(**hyperparams)
            model_cv.fit(train_x, train_y)
            fpr, tpr, _ = roc_curve(valid_y, model_cv.predict_proba(valid_x)[:, 1])
            auroc = auc(fpr, tpr)
            scores.append(auroc)
            print(f"Internal AUROC CV-{r}: {auroc}")
        self.mean_score_ = np.mean(scores)
        print(f"Average AUROC: {self.mean_score_}")
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

        model_path = os.path.join(model_dir, "model.joblib")
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
        model_path = os.path.join(model_dir, "model.joblib")
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


class LazyRandomForestBinaryClassifier(object):

    def __init__(self,
                 reducer_method: str = None,
                 max_reducer_dim: int = 500,
                 base_test_size: float = 0.25,
                 base_num_splits: int = 3,
                 num_trials: int = 100,
                 timeout: int = 600,
                 min_positive_proportion: float=0.01,
                 max_positive_proportion: float=0.5,
                 min_samples: int=30,
                 max_samples: int=10000,
                 min_positive_samples: int=10,
                 max_num_partitions: int=100,
                 random_state: int = 42):
        self.reducer_method = reducer_method
        self.max_reducer_dim = max_reducer_dim
        self.random_state = random_state
        self.base_test_size = base_test_size
        self.base_num_splits = base_num_splits
        self.base_num_trials = num_trials
        self.base_timeout = timeout
        self.min_positive_proportion = min_positive_proportion
        self.max_positive_proportion = max_positive_proportion
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.min_positive_samples = min_positive_samples
        self.max_num_partitions = max_num_partitions
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

    def fit(self, X=None, h5_file=None, h5_idxs=None, y=None):
        iu = InputUtils()
        su = SamplingUtils()
        iu.evaluate_input(X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=y, is_y_mandatory=True)
        X, h5_file, h5_idxs = iu.preprocessing(X=X, h5_file=h5_file, h5_idxs=h5_idxs)
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
                                             max_num_partitions=self.max_num_partitions):
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
            model = BaseRandomForestBinaryClassifier(num_splits=self.base_num_splits, test_size=self.base_test_size, num_trials=self.base_num_trials, timeout = self.base_timeout, random_state=self.random_state)
            model.fit(X_sampled, y_sampled)
            print("Model fitted.")
            reducers += [reducer_]
            models += [model]
        self.reducers = reducers
        self.models = models
        return self

    def predict(self, X=None, h5_file=None, h5_idxs=None, chunk_size=1000):
        iu = InputUtils()
        iu.evaluate_input(X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=None, is_y_mandatory=False)
        X, h5_file, h5_idxs = iu.preprocessing(X=X, h5_file=h5_file, h5_idxs=h5_idxs)
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
            "reducer_method": self.reducer_method,
            "max_reducer_dim": self.max_reducer_dim,
            "random_state": self.random_state,
            "base_test_size": self.base_test_size,
            "base_num_splits": self.base_num_splits,
            "base_num_trials": self.base_num_trials,
            "base_timeout": self.base_timeout
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
        obj.reducer_method = metadata.get("reducer_method", None)
        obj.max_reducer_dim = metadata.get("max_reducer_dim", None)
        obj.random_state = metadata.get("random_state", None)
        obj.base_test_size = metadata.get("base_test_size", None)
        obj.base_num_splits = metadata.get("base_num_splits", None)
        obj.base_num_trials = metadata.get("base_num_trials", None)
        obj.base_timeout = metadata.get("base_timeout", None)
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
    

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification

    num_trials=2
    timeout=10
    print("Creating a synthetic dataset for testing...")
    X, y = make_classification(
        n_samples=10000, 
        n_features=20, 
        n_informative=10, 
        n_redundant=5, 
        weights=[0.99, 0.01],
        random_state=42
    )
    """
    print("Testing with in-memory data...")
    clf = LazyRandomForestBinaryClassifier(reducer_method='pca', max_reducer_dim=10, num_trials=num_trials, timeout=timeout, max_samples=1000, max_num_partitions=3)
    clf.fit(X=X, y=y)
    print("Saving model and loading it")
    clf.save_model("test_model")
    clf_loaded = LazyRandomForestBinaryClassifier.load_model("test_model")
    predictions_loaded = clf_loaded.predict(X)
    print("Loaded Predictions:", predictions_loaded[:10])
    print("Statistics of loaded predictions:")
    print(f"Min: {np.min(predictions_loaded):.4f}")
    print(f"Max: {np.max(predictions_loaded):.4f}")
    print(f"Average: {np.mean(predictions_loaded):.4f}")
    roc_auc_loaded = roc_auc_score(y, predictions_loaded)
    print("Loaded ROC AUC:", roc_auc_loaded)
    print("Test completed successfully.")
    shutil.rmtree("test_model")
    """

    print("Doing a dummy test with h5 file")
    X_ = np.zeros((X.shape[0]*2, X.shape[1]), dtype=X.dtype)
    h5_idxs = random.sample(list(range(X.shape[0])), len(y))
    for i, idx in enumerate(h5_idxs):
        X_[idx, :] = X[i, :]
    with h5py.File("test_data.h5", "w") as f:
        f.create_dataset("values", data=X_)
    model = LazyRandomForestBinaryClassifier(reducer_method='pca', max_reducer_dim=10, num_trials=num_trials, timeout=timeout, max_samples=10000, max_num_partitions=3)
    model.fit(h5_file="test_data.h5", h5_idxs=h5_idxs, y=y)
    model.save_model("test_model")
    model_loaded = LazyRandomForestBinaryClassifier.load_model("test_model")
    predictions_loaded = model_loaded.predict(h5_file="test_data.h5", h5_idxs=h5_idxs)
    print("Loaded Predictions from H5:", predictions_loaded[:10])
    print("Statistics of loaded predictions:")
    print(f"Min: {np.min(predictions_loaded):.4f}")
    print(f"Max: {np.max(predictions_loaded):.4f}")
    print(f"Average: {np.mean(predictions_loaded):.4f}")
    roc_auc_loaded = roc_auc_score(y, predictions_loaded)
    print("Loaded ROC AUC from H5:", roc_auc_loaded)
    print("H5 test completed successfully.")
    shutil.rmtree("test_model")
    os.remove("test_data.h5")