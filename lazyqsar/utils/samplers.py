import math
import random
import h5py
import numpy as np

from .io import InputUtils
from .logging import logger


def compute_head_weights(scores: list, n_samples: int) -> np.ndarray:
    """Blend CV-based weights with uniform weights; shrinkage toward CV grows with n_samples."""
    scores = np.array(scores, dtype=float)
    N = len(scores)
    alpha = n_samples / (n_samples + 500) if n_samples else 0.0
    cv = np.clip(scores - 0.5, 0, 1) + 1e-4
    cv = cv / cv.sum()
    uniform = np.ones(N) / N
    weights = alpha * cv + (1 - alpha) * uniform
    return weights / weights.sum()


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
