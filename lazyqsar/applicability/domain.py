"""
Applicability domain estimator based on Mahalanobis distance in PCA space.

The score is the empirical survival function of the training Mahalanobis
distances — i.e. the fraction of training samples that are *further* from
the centroid than the query:

    score(x) = P(d_train > d(x))   ∈ [0, 1]

score = 1  →  query is closer to the centroid than every training sample
             (fully within the training distribution)
score = 0  →  query is further than every training sample
             (completely out-of-distribution)

Design constraints
------------------
* No KNN at inference (O(n) per query is too slow for large training sets).
* ONNX-only at inference: the fitted model is exported as a single ONNX graph
  built from primitive ops (Sub, MatMul, Mul, Relu, Sqrt, Reshape, Less, Cast,
  Div).  At inference only onnxruntime + numpy are required.
* Works for any X matrix: dense or sparse, binary fingerprints or continuous.

Implementation notes
--------------------
PCA pre-projection is required because:
  (a) raw fingerprints (p=2048) make the p×p covariance matrix huge and
      singular without extensive regularization;
  (b) PCA removes redundant/zero-variance directions, making the distance
      well-conditioned in the retained k-dimensional subspace.

n_components is chosen automatically as min(50, p, n_samples-1) unless
overridden.  The covariance matrix is Tikhonov-regularised before inversion.

Calibration uses 200 evenly-spaced quantiles of the training distances stored
as a lookup table.  The ONNX graph broadcasts each query distance against all
calibration knots and averages the comparisons, implementing the survival
function in a vectorised, loop-free way.
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.decomposition import PCA


_N_CAL_KNOTS = 200


def _to_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        return X.toarray().astype(np.float64)
    return np.asarray(X, dtype=np.float64)


class ApplicabilityDomain:
    """
    Fits an applicability domain estimator on a training feature matrix.

    Internally applies StandardScaler before PCA so that the AD is independent
    of the downstream model's preprocessor and works on raw descriptor output.

    Parameters
    ----------
    n_components : int or None
        Number of PCA components to retain.  None (default) → min(50, p, n-1).
    """

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X) -> "ApplicabilityDomain":
        X = _to_dense(X)
        n, p = X.shape
        if n < 2:
            raise ValueError(f"Need at least 2 training samples, got {n}.")

        # StandardScaler — fitted independently, not shared with any model
        self.scaler_mean_ = X.mean(axis=0).astype(np.float32)  # (p,)
        std = X.std(axis=0)
        std[std < 1e-12] = 1.0  # avoid /0
        self.scaler_scale_ = std.astype(np.float32)  # (p,)
        X = (X - self.scaler_mean_) / self.scaler_scale_

        k = (
            self.n_components
            if self.n_components is not None
            else min(max(1, int(p**0.5)), 100, p, n - 1)
        )
        k = max(1, k)

        self.pca_ = PCA(n_components=k, whiten=False)
        X_pca = self.pca_.fit_transform(X)  # (n, k)

        self.centroid_ = X_pca.mean(axis=0)  # (k,)

        cov = np.cov(X_pca.T) if k > 1 else np.var(X_pca, axis=0).reshape(1, 1)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        eps = max(1e-12, 1e-6 * float(np.trace(cov)) / k)
        cov += np.eye(k) * eps
        try:
            self.precision_ = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.precision_ = np.linalg.pinv(cov)

        train_dists = self._mahal(X_pca)  # (n,)
        self.cal_knots_ = np.quantile(
            train_dists, np.linspace(0.0, 1.0, _N_CAL_KNOTS)
        ).astype(np.float32)

        return self

    # ------------------------------------------------------------------
    # Numpy inference (available before ONNX export)
    # ------------------------------------------------------------------

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.scaler_mean_) / self.scaler_scale_

    def _mahal(self, X_pca: np.ndarray) -> np.ndarray:
        """Mahalanobis distances from centroid_ for already-projected X."""
        delta = X_pca - self.centroid_  # (n, k)
        P_delta = delta @ self.precision_  # (n, k)
        dist_sq = (P_delta * delta).sum(axis=1)  # (n,)
        return np.sqrt(np.clip(dist_sq, 0.0, None))

    def score(self, X) -> np.ndarray:
        """
        Return AD scores in [0, 1] for each row of X.

        Accepts raw descriptor output — scaling is applied internally.
        Uses numpy — works before/without ONNX export.
        """
        X = _to_dense(X)
        X = self._scale(X)
        X_pca = self.pca_.transform(X)
        dists = self._mahal(X_pca)
        # survival function: fraction of calibration knots > query dist
        scores = (dists[:, None] < self.cal_knots_[None, :]).mean(axis=1)
        return scores.astype(np.float32)

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def to_onnx(self, path: str) -> None:
        """
        Build and save an ONNX graph for the AD estimator.

        Graph inputs:  X  (float32, shape [batch, n_features_in])
        Graph outputs: score (float32, shape [batch])

        All arithmetic is primitive ONNX ops; no custom extensions needed.
        """
        import onnx
        from onnx import helper, TensorProto, numpy_helper

        scaler_mean = self.scaler_mean_.astype(np.float32)  # (p,)
        scaler_scale = self.scaler_scale_.astype(np.float32)  # (p,)
        pca_mean = self.pca_.mean_.astype(np.float32)  # (p,)
        # store components transposed for MatMul: (p, k)
        pca_comp_T = self.pca_.components_.T.astype(np.float32)  # (p, k)
        centroid = self.centroid_.astype(np.float32)  # (k,)
        precision = self.precision_.astype(np.float32)  # (k, k)
        cal_knots = self.cal_knots_.astype(np.float32)  # (N_cal,)

        p = pca_mean.shape[0]
        k = centroid.shape[0]
        N_cal = cal_knots.shape[0]

        def init(name, arr):
            return numpy_helper.from_array(arr, name=name)

        initializers = [
            init("scaler_mean", scaler_mean),
            init("scaler_scale", scaler_scale),
            init("pca_mean", pca_mean),
            init("pca_comp_T", pca_comp_T),
            init("centroid", centroid),
            init("precision", precision),
            init("cal_knots", cal_knots),
            init("ones_k", np.ones((k, 1), dtype=np.float32)),
            init("ones_cal", np.ones((N_cal, 1), dtype=np.float32)),
            init("n_cal_f", np.array(N_cal, dtype=np.float32)),
            init("shape_B", np.array([-1], dtype=np.int64)),
            init("shape_B1", np.array([-1, 1], dtype=np.int64)),
            init("shape_1N", np.array([1, N_cal], dtype=np.int64)),
        ]

        nodes = [
            # ── StandardScaler ──────────────────────────────────────
            # x_sc = (X - scaler_mean) / scaler_scale   [B, p]
            helper.make_node("Sub", ["X", "scaler_mean"], ["x_centered"]),
            helper.make_node("Div", ["x_centered", "scaler_scale"], ["x_sc"]),
            # ── PCA projection ──────────────────────────────────────
            # x_c = x_sc - pca_mean                 [B, p]
            helper.make_node("Sub", ["x_sc", "pca_mean"], ["x_c"]),
            # x_pca = x_c @ pca_comp_T             [B, k]
            helper.make_node("MatMul", ["x_c", "pca_comp_T"], ["x_pca"]),
            # ── Mahalanobis distance ─────────────────────────────────
            # delta = x_pca - centroid              [B, k]
            helper.make_node("Sub", ["x_pca", "centroid"], ["delta"]),
            # P_delta = delta @ precision           [B, k]
            helper.make_node("MatMul", ["delta", "precision"], ["P_delta"]),
            # elem = P_delta * delta  (elementwise) [B, k]
            helper.make_node("Mul", ["P_delta", "delta"], ["elem"]),
            # dist_sq_2d = elem @ ones_k            [B, 1]
            helper.make_node("MatMul", ["elem", "ones_k"], ["dist_sq_2d"]),
            # dist_sq = reshape → [B]
            helper.make_node("Reshape", ["dist_sq_2d", "shape_B"], ["dist_sq"]),
            # clip negatives (numerical noise)
            helper.make_node("Relu", ["dist_sq"], ["dist_sq_nn"]),
            # dist = sqrt(dist_sq_nn)               [B]
            helper.make_node("Sqrt", ["dist_sq_nn"], ["dist"]),
            # ── Calibration (survival function) ─────────────────────
            # dist_2d  = reshape dist  → [B, 1]
            helper.make_node("Reshape", ["dist", "shape_B1"], ["dist_2d"]),
            # cal_2d   = reshape knots → [1, N_cal]
            helper.make_node("Reshape", ["cal_knots", "shape_1N"], ["cal_2d"]),
            # in_domain = dist_2d < cal_2d          [B, N_cal] bool
            helper.make_node("Less", ["dist_2d", "cal_2d"], ["in_domain_bool"]),
            # cast to float32
            helper.make_node(
                "Cast", ["in_domain_bool"], ["in_domain"], to=TensorProto.FLOAT
            ),
            # sum over calibration axis: in_domain @ ones_cal → [B, 1]
            helper.make_node("MatMul", ["in_domain", "ones_cal"], ["sum_2d"]),
            # reshape → [B]
            helper.make_node("Reshape", ["sum_2d", "shape_B"], ["sum_1d"]),
            # score = sum / N_cal                   [B]
            helper.make_node("Div", ["sum_1d", "n_cal_f"], ["score"]),
        ]

        X_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, p])
        score_out = helper.make_tensor_value_info("score", TensorProto.FLOAT, [None])

        graph = helper.make_graph(
            nodes, "applicability_domain", [X_input], [score_out], initializers
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7

        onnx.checker.check_model(model)
        with open(path, "wb") as fh:
            fh.write(model.SerializeToString())

    def save(self, directory: str) -> None:
        """Save ONNX model + metadata JSON to *directory*."""
        os.makedirs(directory, exist_ok=True)
        self.to_onnx(os.path.join(directory, "applicability_domain.onnx"))
        meta = {
            "n_components": int(self.pca_.n_components_),
            "n_features_in": int(self.pca_.n_features_in_),
            "n_cal_knots": int(len(self.cal_knots_)),
            "cal_min": float(self.cal_knots_[0]),
            "cal_max": float(self.cal_knots_[-1]),
            "scaler": "standard",
        }
        with open(os.path.join(directory, "applicability_domain.json"), "w") as fh:
            json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# Inference-only artifact (onnxruntime only)
# ---------------------------------------------------------------------------


class ApplicabilityDomainArtifact:
    """
    Inference-only applicability domain loaded from a saved ONNX model.

    Requires only onnxruntime and numpy — no sklearn, no scipy.
    """

    def __init__(self) -> None:
        self._session = None
        self.metadata: dict = {}

    @classmethod
    def load(cls, directory: str) -> "ApplicabilityDomainArtifact":
        inst = cls()
        json_path = os.path.join(directory, "applicability_domain.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"No AD metadata found at {json_path!r}")
        with open(json_path) as fh:
            inst.metadata = json.load(fh)

        import onnxruntime as rt

        onnx_path = os.path.join(directory, "applicability_domain.onnx")
        inst._session = rt.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        return inst

    def score(self, X) -> np.ndarray:
        """
        Return AD scores in [0, 1] for each row of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix — same featurizer as used during fit.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,), dtype float32
            1.0 = fully in-domain, 0.0 = fully out-of-domain.
        """
        if hasattr(X, "toarray"):
            X = X.toarray()
        X_f32 = np.asarray(X, dtype=np.float32)
        input_name = self._session.get_inputs()[0].name
        return self._session.run(None, {input_name: X_f32})[0]
