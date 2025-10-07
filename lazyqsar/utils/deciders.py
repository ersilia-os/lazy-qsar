import numpy as np
import logging

from .samplers import BinaryClassifierSamplingUtils
from .evaluators import QuickAUCEstimator
from .logging import logger

logging.getLogger("joblib").setLevel(logging.CRITICAL)


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
        self.logger = logger
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
        self.logger.debug(
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
