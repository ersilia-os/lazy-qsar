import json
import joblib
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

import onnx
from onnx import helper, numpy_helper, TensorProto

from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION
from ...utils.logging import logger
from . import search_cv_splits


MAX_ITER = 1000
LINEARSVC_CONFIGS = [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}]
SGD_CONFIGS = [{"alpha": 0.001}, {"alpha": 0.01}, {"alpha": 0.1}]
SPARSITY_THRESHOLD = 0.9


def _is_sparse(X):
    if hasattr(X, "toarray"):
        zero_fraction = 1.0 - (X.count_nonzero() / np.prod(X.shape))
    else:
        zero_fraction = np.mean(X == 0)
    return zero_fraction >= SPARSITY_THRESHOLD


def use_full(X):
    if X.shape[1] <= 512:
        return True
    else:
        return _is_sparse(X)


def find_params(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    do_full = use_full(X)
    cv = StratifiedShuffleSplit(n_splits=search_cv_splits(len(y)), test_size=0.2, random_state=42)

    if do_full:
        logger.info(f"Evaluating {len(LINEARSVC_CONFIGS)} LinearSVC configs...")

        def eval_config(cfg):
            scores = []
            for tr, va in cv.split(X, y):
                clf = LinearSVC(
                    C=cfg["C"],
                    class_weight="balanced",
                    max_iter=MAX_ITER,
                    random_state=42,
                )
                clf.fit(X[tr], y[tr])
                scores.append(roc_auc_score(y[va], clf.decision_function(X[va]).astype(np.float32)))
            return float(np.mean(scores))

        with ThreadPoolExecutor(max_workers=len(LINEARSVC_CONFIGS)) as ex:
            results = list(ex.map(eval_config, LINEARSVC_CONFIGS))

        best_idx = int(np.argmax(results))
        best_C = LINEARSVC_CONFIGS[best_idx]["C"]
        cv_score = results[best_idx]
        logger.info(f"Best LinearSVC C: {best_C} (AUC: {cv_score:.4f})")
        return {"C": best_C, "use_linearsvc": True, "cv_score": cv_score}

    else:
        logger.info(f"Evaluating {len(SGD_CONFIGS)} SGD hinge configs...")

        def eval_config(cfg):
            scores = []
            for tr, va in cv.split(X, y):
                clf = SGDClassifier(
                    loss="hinge",
                    alpha=cfg["alpha"],
                    class_weight="balanced",
                    max_iter=MAX_ITER,
                    n_jobs=-1,
                    random_state=42,
                )
                clf.fit(X[tr], y[tr])
                scores.append(roc_auc_score(y[va], clf.decision_function(X[va]).astype(np.float32)))
            return float(np.mean(scores))

        with ThreadPoolExecutor(max_workers=len(SGD_CONFIGS)) as ex:
            results = list(ex.map(eval_config, SGD_CONFIGS))

        best_idx = int(np.argmax(results))
        best_alpha = SGD_CONFIGS[best_idx]["alpha"]
        cv_score = results[best_idx]
        logger.info(f"Best SGD hinge alpha: {best_alpha} (AUC: {cv_score:.4f})")
        return {"alpha": best_alpha, "use_linearsvc": False, "cv_score": cv_score}


class Head(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=None, C=None, use_linearsvc=False, cv_score=None):
        self.alpha = alpha
        self.C = C
        self.use_linearsvc = use_linearsvc
        self.cv_score = cv_score

    def _fit(self, X, y):
        X = np.asarray(X)
        if self.use_linearsvc:
            logger.info("Fitting LinearSVC head...")
            self.model = LinearSVC(
                C=self.C or 1.0,
                class_weight="balanced",
                max_iter=MAX_ITER,
                random_state=42,
            )
        else:
            logger.info("Fitting SGD hinge head...")
            self.model = SGDClassifier(
                loss="hinge",
                alpha=self.alpha,
                class_weight="balanced",
                max_iter=MAX_ITER,
                n_jobs=-1,
                random_state=42,
            )
        self.model.fit(X, y)
        self.input_dim = X.shape[1]
        return self

    def _calibrate(self, X, y):
        """Platt scaling: fit a LR calibrator on OOF decision function values."""
        splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        oof_df, oof_y = [], []
        for tr, va in splitter.split(X, y):
            self._fit(X[tr], y[tr])
            oof_df.extend(self.model.decision_function(X[va]).astype(np.float32))
            oof_y.extend(y[va])
        oof_df = np.array(oof_df).reshape(-1, 1)
        oof_y = np.array(oof_y)
        self.calibrator = LogisticRegression(solver="lbfgs", class_weight="balanced")
        self.calibrator.fit(oof_df, oof_y)
        logger.info(f"SVC calibrator fitted on {len(oof_y)} OOF samples.")

    def fit(self, X, y):
        self._calibrate(np.asarray(X), np.asarray(y))
        self._fit(np.asarray(X), y)
        self.score = self.cv_score if self.cv_score is not None else 0.5
        logger.info(f"SVC head score (from CV): {self.score:.4f}")
        return self

    def predict_proba(self, X):
        df = self.model.decision_function(np.asarray(X)).astype(np.float32)
        p = self.calibrator.predict_proba(df.reshape(-1, 1))[:, 1]
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def save(self, name, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        metadata = {
            "alpha": self.alpha,
            "C": self.C,
            "score": self.score,
            "input_dim": self.input_dim,
            "use_linearsvc": bool(self.use_linearsvc),
            "cv_score": self.cv_score,
        }
        with open(os.path.join(model_dir, f"{name}_metadata.json"), "w") as f:
            json.dump(metadata, f)
        joblib.dump(self.model, os.path.join(model_dir, f"{name}_model.joblib"))
        joblib.dump(self.calibrator, os.path.join(model_dir, f"{name}_calibrator.joblib"))

    @classmethod
    def load(cls, name, model_dir):
        with open(os.path.join(model_dir, f"{name}_metadata.json"), "r") as f:
            metadata = json.load(f)
        model = joblib.load(os.path.join(model_dir, f"{name}_model.joblib"))
        head = cls(
            alpha=metadata.get("alpha"),
            C=metadata.get("C"),
            use_linearsvc=metadata.get("use_linearsvc", False),
            cv_score=metadata.get("cv_score"),
        )
        head.model = model
        head.calibrator = joblib.load(os.path.join(model_dir, f"{name}_calibrator.joblib"))
        head.score = metadata["score"]
        head.input_dim = metadata["input_dim"]
        return head


def convert_to_onnx(name: str, model_dir: str) -> str:
    """
    Convert to ONNX: p = σ(a*(w^T x + b) + c)
    where (a, c) are Platt calibrator coefficients.
    Folded into a single GEMM: w_cal = a*w, b_cal = a*b + c.
    """
    head = Head.load(name, model_dir)
    clf = head.model
    cal = head.calibrator
    input_dim = int(head.input_dim)

    a = float(cal.coef_[0][0])
    c = float(cal.intercept_[0])

    w = np.asarray(clf.coef_, dtype=np.float32).reshape(1, input_dim)
    b = np.asarray(clf.intercept_, dtype=np.float32).reshape(1,)

    w = (a * w).astype(np.float32)
    b = (a * b + c).astype(np.float32)

    W_init = numpy_helper.from_array(w.T, name=f"W_{name}")
    b_init = numpy_helper.from_array(b, name=f"b_{name}")
    shape1d_init = numpy_helper.from_array(
        np.array([-1], np.int64), name=f"shape_out_{name}"
    )

    X = helper.make_tensor_value_info(
        f"input_{name}", TensorProto.FLOAT, ["batch_size", input_dim]
    )
    Y = helper.make_tensor_value_info(
        f"output_{name}", TensorProto.FLOAT, ["batch_size"]
    )

    gemm = helper.make_node(
        "Gemm",
        inputs=[f"input_{name}", f"W_{name}", f"b_{name}"],
        outputs=[f"logit_{name}"],
        name=f"{name}_Gemm",
    )
    sigm = helper.make_node(
        "Sigmoid",
        inputs=[f"logit_{name}"],
        outputs=[f"prob_{name}"],
        name=f"{name}_Sigmoid",
    )
    reshape = helper.make_node(
        "Reshape",
        inputs=[f"prob_{name}", f"shape_out_{name}"],
        outputs=[f"output_{name}"],
        name=f"{name}_Reshape1D",
    )

    graph = helper.make_graph(
        nodes=[gemm, sigm, reshape],
        name=f"{name}",
        inputs=[X],
        outputs=[Y],
        initializer=[W_init, b_init, shape1d_init],
    )

    model = helper.make_model(
        graph,
        producer_name=f"{name}",
        opset_imports=[helper.make_operatorsetid("", ONNX_TARGET_OPSET)],
    )
    model.ir_version = ONNX_IR_VERSION

    onnx.checker.check_model(model)
    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    onnx.save(model, onnx_path)
    return onnx_path
