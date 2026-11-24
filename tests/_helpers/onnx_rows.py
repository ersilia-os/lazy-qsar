"""Find training rows stored inside a checkpoint's ONNX graphs.

A kernel SVM's graph carries its support vectors, which are rows of the (preprocessed)
training matrix. This helper looks for such rows in every float initializer and float
attribute of every ``.onnx`` file under a directory, in either orientation: skl2onnx's
``SVMClassifier`` keeps them row-major in an attribute list, LazyQSAR's own RBF SVR graph
keeps them transposed in an initializer.
"""

from __future__ import annotations

import os

import numpy as np


def _float_arrays(model):
    from onnx import numpy_helper

    for tensor in model.graph.initializer:
        arr = numpy_helper.to_array(tensor)
        if np.issubdtype(arr.dtype, np.floating):
            yield arr
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.floats:
                yield np.asarray(attr.floats, dtype=np.float32)


def _candidate_rows(arr, n_features):
    if arr.ndim == 2 and arr.shape[0] == n_features and arr.shape[1] != n_features:
        yield arr.T
    flat = arr.ravel()
    if flat.size and flat.size % n_features == 0:
        yield flat.reshape(-1, n_features)


def training_rows_in_onnx(directory, X, rtol=1e-5, atol=1e-6) -> int:
    """Count the distinct rows of *X* found verbatim in any ``.onnx`` under *directory*.

    Rows are compared at float32 precision, which is what a graph stores.
    """
    import onnx

    X = np.asarray(X, dtype=np.float32)
    n_features = X.shape[1]
    found = np.zeros(len(X), dtype=bool)
    for root, _, files in os.walk(directory):
        for name in files:
            if not name.endswith(".onnx"):
                continue
            model = onnx.load(os.path.join(root, name))
            for arr in _float_arrays(model):
                for rows in _candidate_rows(
                    np.asarray(arr, dtype=np.float32), n_features
                ):
                    for row in rows:
                        match = np.all(np.isclose(X, row, rtol=rtol, atol=atol), axis=1)
                        found |= match
    return int(found.sum())


def head_files(directory) -> set:
    """Names of every head graph saved under *directory* (``svc.onnx``, ``xgboost.onnx``...)."""
    return {
        name
        for _, _, files in os.walk(directory)
        for name in files
        if name.endswith(".onnx")
    }
