import math
import random
import h5py
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from .io import InputUtils
from .logging import logger


class BinaryClassifierSamplingUtils(object):
    def chunk_h5_file(self, h5_file, h5_idxs, chunk_size):
        iu = InputUtils()
        with h5py.File(h5_file, "r") as f:
            keys = f.keys()
            if "values" in keys:
                values_key = "values"
            elif "Values" in keys:
                values_key = "Values"
            else:
                raise Exception("HDF5 does not contain a values key")
            values = f[values_key]
            for i in range(0, len(h5_idxs), chunk_size):
                idxs_chunk = h5_idxs[i : i + chunk_size]
                yield iu.h5_data_reader(values, idxs_chunk)

    def chunk_matrix(self, X, chunk_size):
        for i in range(0, X.shape[0], chunk_size):
            yield X[i : i + chunk_size]

    def get_partition_indices(
        self,
        X,
        h5_file,
        h5_idxs,
        y,
        max_num_partitions=100,
        max_samples=100_000,
    ):
        iu = InputUtils()
        iu.evaluate_input(
            X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=y, is_y_mandatory=True
        )
        n_tot = len(y)
        logger.info(
            "Dataset summary:\n"
            f"  • Total samples: {n_tot}\n"
            f"  • Positive samples: {sum(y)}\n"
            f"  • Negative samples: {n_tot - sum(y)}\n"
            f"  • Max samples per partition: {max_samples}"
        )

        # If dataset fits within one partition, yield everything and stop
        if n_tot <= max_samples:
            logger.info("Generating 1 partition (full dataset).")
            yield list(range(n_tot))
            return

        # Shuffle indices and slice cyclically into non-overlapping chunks
        idxs = list(range(n_tot))
        random.shuffle(idxs)
        n_partitions = min(max_num_partitions, math.ceil(n_tot / max_samples))
        logger.info(f"Generating {n_partitions} partitions of {max_samples} samples each.")

        for i in range(n_partitions):
            start = (i * max_samples) % n_tot
            end = start + max_samples
            if end <= n_tot:
                chunk = idxs[start:end]
            else:
                chunk = idxs[start:] + idxs[:end - n_tot]
            yield sorted(chunk)


class KFolder(object):
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y, groups=None):
        skf = KFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )
        for train_idxs, test_idxs in skf.split(X, y):
            yield train_idxs, test_idxs


class StratifiedKFolder(object):
    def __init__(
        self,
        test_size=0.25,
        n_splits=5,
        max_positive_proportion=0.5,
        shuffle=True,
        random_state=None,
    ):
        self.test_size = test_size
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.max_positive_proportion = max_positive_proportion

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y, groups=None):
        num_splits = max(3, int(1 / self.test_size))
        splitter = StratifiedKFold(
            n_splits=num_splits, shuffle=self.shuffle, random_state=self.random_state
        )
        done_folds = 0
        for train_idxs, test_idxs in splitter.split(X, y):
            train_idxs_pos = [i for i in train_idxs if y[i] == 1]
            train_idxs_neg = [i for i in train_idxs if y[i] == 0]
            if len(train_idxs_pos) / len(train_idxs) > self.max_positive_proportion:
                expected_neg = len(train_idxs) * (1 - self.max_positive_proportion)
                n_missing = int(expected_neg - len(train_idxs_neg))
                if n_missing > 0:
                    additional_neg_idxs = random.choices(train_idxs_neg, k=n_missing)
                    train_idxs = list(train_idxs) + additional_neg_idxs
                    random.shuffle(train_idxs)
            done_folds += 1
            if done_folds >= self.n_splits:
                break
            yield train_idxs, test_idxs
