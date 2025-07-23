import numpy as np
from tqdm import tqdm
import multiprocessing
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from .samplers import BinaryClassifierSamplingUtils
from .samplers import StratifiedKFolder
from .evaluators import QuickAUCEstimator


NUM_CPU = max(1, int(multiprocessing.cpu_count() / 2))

import logging

logging.getLogger("joblib").setLevel(logging.CRITICAL)


class BinaryClassifierPCADecider(object):
    def __init__(self, X, y, max_positive_proportion=0.5):
        self.X = X
        self.y = y
        self.max_positive_proportion = max_positive_proportion
        self.timeout = 600
        self.components_to_evaluate = [0.9]
        self.num_evidence_in_folds = 3

    def decide(self):
        start_time = time.time()
        print("Deciding whether to use PCA for the binary classifier.")
        cv = StratifiedKFolder(
            test_size=0.25,
            n_splits=5,
            max_positive_proportion=self.max_positive_proportion,
            shuffle=True,
            random_state=42,
        )
        pca_scores = []
        for n_components in tqdm(
            self.components_to_evaluate, desc="Evaluating PCA components"
        ):
            current_time = time.time()
            if current_time - start_time > self.timeout:
                return True
            scores = []
            for train_idxs, test_idxs in cv.split(self.X, self.y):
                X_train = self.X[train_idxs]
                y_train = self.y[train_idxs]
                X_test = self.X[test_idxs]
                scaler = StandardScaler()
                pca = PCA(n_components=n_components)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)
                X_train_pca = scaler.fit_transform(X_train_pca)
                X_test_pca = scaler.transform(X_test_pca)
                cv = StratifiedKFold()
                model = LogisticRegressionCV(
                    cv=cv, class_weight="balanced", n_jobs=NUM_CPU
                )
                model.fit(X_train_pca, y_train)
                y_pred = model.predict_proba(X_test_pca)[:, 1]
                auc_score = roc_auc_score(self.y[test_idxs], y_pred)
                scores += [auc_score]
                if len(scores) > self.num_evidence_in_folds:
                    break
            pca_scores += [np.mean(scores)]
        pca_score = np.mean(pca_scores)

        scores = []
        for train_idxs, test_idxs in tqdm(
            cv.split(self.X, self.y), desc="Evaluating without PCA"
        ):
            current_time = time.time()
            if current_time - start_time > self.timeout:
                return True
            X_train = self.X[train_idxs]
            y_train = self.y[train_idxs]
            X_test = self.X[test_idxs]
            model = LogisticRegressionCV(cv=cv, class_weight="balanced", n_jobs=NUM_CPU)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(self.y[test_idxs], y_pred)
            scores += [auc_score]
            if len(scores) > self.num_evidence_in_folds:
                break
        no_pca_score = np.mean(scores)

        print(f"Best PCA score: {pca_score:.4f}, No PCA score: {no_pca_score:.4f}")

        if abs(pca_score - no_pca_score) < 0.02:
            return True

        if pca_score > no_pca_score:
            return True
        else:
            return False


class BinaryClassifierMaxSamplesDecider(object):
    def __init__(
        self,
        X,
        y,
        min_samples,
        min_positive_proportion,
        max_steps=5,
        absolute_min=10000,
        absolute_max=100000,
    ):
        self.X = X
        self.y = y
        self.min_positive_proportion = min_positive_proportion
        self.min_samples = min_samples
        self.max_steps = max_steps
        self.absolute_max = absolute_max
        self.absolute_min = absolute_min
        self.quick_auc_estimator = QuickAUCEstimator()

    def _get_min_and_max_range(self):
        return self.absolute_min, self.absolute_max

    def _generate_steps(self, n_min, n_max):
        if n_max <= n_min:
            return [n_min]
        range_size = n_max - n_min + 1
        num_steps = min(self.max_steps, range_size)
        step = max(1, (n_max - n_min) // (num_steps - 1)) if num_steps > 1 else 1
        result = list(range(n_min, n_max + 1, step))
        if result[-1] != n_max:
            result.append(n_max)
        return result

    def decide(self):
        print(
            "Quickly deciding the max number of samples to use for the binary classifier."
        )
        n_min, n_max = self._get_min_and_max_range()
        if self.X.shape[0] < n_min:
            return n_min
        n_samples = []
        scores = []
        for n in self._generate_steps(n_min, n_max):
            auc_scores = []
            c = 0
            for idxs in BinaryClassifierSamplingUtils().get_partition_indices(
                X=self.X,
                h5_file=None,
                h5_idxs=None,
                y=self.y,
                min_positive_proportion=self.min_positive_proportion,
                max_positive_proportion=0.5,
                min_samples=self.min_samples,
                max_samples=n,
                min_positive_samples=10,
                max_num_partitions=100,
                min_seen_across_partitions=1,
            ):
                X_sampled = self.X[idxs, :]
                y_sampled = self.y[idxs]
                auc_score = self.quick_auc_estimator.estimate(X_sampled, y_sampled)
                auc_scores += [auc_score]
                c += 1
                if c >= 3:
                    break
            scores += [np.mean(auc_scores)]
            n_samples += [n]
        return n_samples[np.argmax(scores)]
