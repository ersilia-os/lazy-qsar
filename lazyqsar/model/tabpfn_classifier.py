from tabpfn import TabPFNClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from .rf_pfn import RandomForestTabPFNClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import random


NUM_FEATURES = 500


def fit_feature_reducer(X, y, method="best", max_dim=None):
    if max_dim is None:
        max_dim = NUM_FEATURES
    if X.shape[0] > max_dim:
        return []
    reducer_0 = VarianceThreshold(threshold=0)
    reducer_0.fit(X)
    X = reducer_0.transform(X)
    if X.shape[0] > max_dim:
        return [reducer_0]
    if method == "pca":
        reducer_1 = PCA(n_components=max_dim)
        reducer_1.fit(X)
        X = reducer_1.transform(X)
        return [reducer_0, reducer_1]
    elif method == "best":
        reducer_1 = SelectKBest(f_classif, k=max_dim)
        reducer_1.fit(X, y)
        X = reducer_1.transform(X)
        return [reducer_0, reducer_1]
    else:
        raise Exception("Wrong feature reduction method. Use 'pca' or 'best'.")


def sliding_chunks(data, chunk_size, max_chunks=100):
    n = len(data)
    if n > chunk_size:
        raise Exception("Chunksize is too big")
    


def sliding_chunks(data, chunk_size=10000, step=1000, max_chunks=None):
    chunks = []
    n = len(data)

    if max_chunks is not None:
        max_possible_start = max(0, n - chunk_size)
        if max_possible_start // step + 1 > max_chunks:
            step = max(1, max_possible_start // (max_chunks - 1))  # adjust step

    for start in range(0, n - chunk_size + 1, step):
        end = start + chunk_size
        chunks.append(data[start:end])
        if max_chunks is not None and len(chunks) >= max_chunks:
            break

    return chunks


def get_partition_indexes(X, y, min_ratio=0.1, max_ratio=0.5, min_samples=30, max_samples=10000, min_pos_samples=10, max_num_partitions=100):
    pos_idxs = [i for i, _ in enumerate(y) if y == 1]
    neg_idxs = [i for i, _ in enumerate(y) if y == 0]
    random.shuffle(pos_idxs)
    random.shuffle(neg_idxs)
    n_tot = len(y)
    n_pos = len(pos_idxs)
    n_neg = len(neg_idxs)
    if n_pos < min_pos_samples:
        raise Exception("Not enough positive samples.")
    if len(y) < min_samples:
        raise Exception("Not enough samples.")
    current_ratio = n_pos / n_neg
    if current_ratio <= min_ratio:
        desired_ratio = min_ratio
    elif current_ratio >= max_ratio:
        desired_ratio = max_ratio
    else:
        desired_ratio = current_ratio
    if n_tot <= max_samples:
        if desired_ratio > current_ratio:
            n_neg = int(n_pos / desired_ratio)
            partitions = []
            for part_idxs in sliding_chunks(neg_idxs):
                idxs = pos_idxs + part_idxs
                random.shuffle(idxs)
                partitions += [idxs]
        elif desired_ratio < current_ratio:
            n_pos = int(n_pos * desired_ratio)
        else:
            idxs = pos_idxs + neg_idxs
            random.shuffle(idxs)
            partitions = [idxs]
    else:
        
        total_samples = min(max_samples, len(y))
    

