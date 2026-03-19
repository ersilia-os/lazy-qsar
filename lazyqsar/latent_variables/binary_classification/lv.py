import json
import os


import joblib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from lazyqsar.utils._install_extras import ensure_torch_cpu

try:
    import torch
except ImportError:
    ensure_torch_cpu()
    import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.random_projection import SparseRandomProjection

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import TensorProto

from ...utils.logging import logger
from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION

from scipy import sparse

MIN_FEATURES = 4
MAX_FEATURES = 512


def find_params(X, y):
    logger.info("Finding optimal latent variable parameters...")

    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

    min_n_components = []
    max_n_components = []
    seed_n_components = []

    logger.debug("Preparing folds for cross-validation...")
    folds = []
    for train_index, test_index in cv.split(X, y):
        logger.debug("Precomputing reductions for a fold...")
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        n_components = min(X_tr.shape[1], X_tr.shape[0]) - 1
        n_components = min(n_components, MAX_FEATURES)
        if X_tr.shape[1] < 500:
            svd_solver = "full"
        else:
            if n_components < 0.8 * min(X_tr.shape[0], X_tr.shape[1]):
                svd_solver = "randomized"
            else:
                svd_solver = "full"
        reducer = PCA(n_components=n_components, svd_solver=svd_solver, random_state=42)
        reducer.fit(np.array(X_tr))
        X_tr = reducer.transform(X_tr)
        X_te = reducer.transform(X_te)
        folds += [(X_tr, X_te, y_tr, y_te)]
        explained_variance_ratio_cumsum = np.cumsum(reducer.explained_variance_ratio_)
        n_components_80 = np.searchsorted(explained_variance_ratio_cumsum, 0.8) + 1
        n_components_90 = np.searchsorted(explained_variance_ratio_cumsum, 0.9) + 1
        n_components_99 = np.searchsorted(explained_variance_ratio_cumsum, 0.99) + 1
        min_n_components += [n_components_80]
        seed_n_components += [n_components_90]
        max_n_components += [n_components_99]

    min_n = int(np.mean(min_n_components))
    seed_n = int(np.mean(seed_n_components))
    max_n = int(np.mean(max_n_components))

    n_low = max(MIN_FEATURES, min_n)
    n_mid = seed_n
    n_high = min(MAX_FEATURES, max_n)
    configs = [n_low, n_mid, n_high]
    seen = set()
    configs = [n for n in configs if not (n in seen or seen.add(n))]

    logger.info(f"Evaluating n_components configs: {configs}")

    def eval_config(n):
        clf = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=5,
            random_state=42,
        )
        scores = []
        for X_tr, X_te, y_tr, y_te in folds:
            clf.fit(X_tr[:, :n], y_tr)
            scores.append(roc_auc_score(y_te, clf.predict_proba(X_te[:, :n])[:, 1]))
        return float(np.mean(scores))

    with ThreadPoolExecutor(max_workers=len(configs)) as ex:
        results = list(ex.map(eval_config, configs))

    best_idx = int(np.argmax(results))
    best_n = min(MAX_FEATURES, configs[best_idx])
    logger.info(f"Best n_components: {best_n} (AUC: {results[best_idx]:.4f})")
    return {"n_components": best_n}


class LatentVariables(object):
    """
    Dimensionality reducer for binary classification using SparseRandomProjection.

    The optimal number of components is found via PCA-based explained-variance
    heuristics in find_params(), then a SparseRandomProjection is fitted with
    that many components for fast, memory-efficient inference and ONNX export.

    Parameters
    ----------
    n_components : int or None
        Number of projection components. If None, no reduction is applied.
    """

    def __init__(self, n_components: int = None):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.input_dim = X.shape[1]
        if self.n_components is None:
            self.reducer = None
            return self
        logger.info(
            "Fitting latent reducer with {0} components...".format(self.n_components)
        )
        n_components = min(self.n_components, X.shape[1])
        self.reducer = SparseRandomProjection(
            n_components=n_components, random_state=42
        )
        self.reducer.fit(X)

        return self

    def transform(self, X, y=None):
        if not hasattr(self, "reducer"):
            raise RuntimeError(
                "The reducer has not been fitted yet. Please call 'fit' before 'transform'."
            )
        if self.reducer is None:
            return X
        logger.info("Transforming latent reducer using SparseRandomProjection...")
        X = self.reducer.transform(X)
        return X

    def save(self, name: str, model_dir: str):
        if not os.path.exists(model_dir):
            logger.info(
                f"Creating directory {model_dir} for saving the latent reducer."
            )
            os.makedirs(model_dir)
        metadata = {"n_components": self.n_components, "input_dim": self.input_dim}
        meta_path = os.path.join(model_dir, f"{name}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        reducer_path = os.path.join(model_dir, f"{name}.joblib")
        joblib.dump(self.reducer, reducer_path)

    @classmethod
    def load(cls, name: str, model_dir: str):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"The directory {model_dir} does not exist.")
        meta_path = os.path.join(model_dir, f"{name}_metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"The metadata file {meta_path} does not exist in the directory {model_dir}."
            )
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        obj = cls(n_components=metadata["n_components"])
        obj.input_dim = metadata["input_dim"]
        reducer_path = os.path.join(model_dir, f"{name}.joblib")
        if not os.path.exists(reducer_path):
            raise FileNotFoundError(
                f"The reducer file {reducer_path} does not exist in the directory {model_dir}."
            )
        obj.reducer = joblib.load(reducer_path)
        return obj


class SparseRandomProjectionTorch(nn.Module):
    """Torch implementation of SparseRandomProjection using fixed projection matrix."""

    def __init__(self, W: np.ndarray):
        super().__init__()
        assert W.ndim == 2, "Projection matrix must be 2D"
        self.register_buffer("W", torch.tensor(W, dtype=torch.float32))

    def forward(self, X):
        return torch.matmul(X, self.W.t())


def convert_to_onnx(name: str, model_dir: str) -> str:
    lv = LatentVariables.load(name, model_dir)
    srp = lv.reducer

    if not hasattr(srp, "components_"):
        raise ValueError("SparseRandomProjection must be fitted before conversion.")

    onnx_path = os.path.join(model_dir, f"{name}.onnx")

    W = srp.components_
    if sparse.issparse(W):
        logger.info(
            f"Sparse projection matrix detected ({W.nnz} non-zeros). Densifying for ONNX export."
        )
        W = W.toarray().astype(np.float32)
    else:
        W = np.asarray(W, dtype=np.float32)

    model = SparseRandomProjectionTorch(W)
    model.eval()

    n_features = W.shape[1]
    dummy_input = torch.randn(1, n_features, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=[f"input_{name}"],
        output_names=[f"projected_{name}"],
        dynamic_axes={
            f"input_{name}": {0: "batch_size"},
            f"projected_{name}": {0: "batch_size"},
        },
        opset_version=ONNX_TARGET_OPSET,
    )

    onnx_model = onnx.load(onnx_path)
    output_name = f"output_{name}"

    shape2d = np.array([-1, W.shape[0]], dtype=np.int64)
    onnx_model.graph.initializer.extend(
        [numpy_helper.from_array(shape2d, name=f"shape2d_{name}")]
    )

    reshape_node = helper.make_node(
        "Reshape",
        inputs=[f"projected_{name}", f"shape2d_{name}"],
        outputs=[output_name],
        name=f"Output_Reshape2D_{name}",
    )

    onnx_model.graph.node.extend([reshape_node])

    del onnx_model.graph.output[:]
    onnx_model.graph.output.extend(
        [
            helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, ["batch_size", W.shape[0]]
            )
        ]
    )

    onnx_model.graph.name = f"{name}"
    onnx_model.ir_version = ONNX_IR_VERSION

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_path)

    logger.info(f"SparseRandomProjection ONNX model saved to {onnx_path} (input: {n_features} -> output: {W.shape[0]})")

    return onnx_path
