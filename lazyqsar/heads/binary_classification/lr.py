import json
import joblib
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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
MAX_ITER_CV = 200  # sufficient for relative ranking during hyperparameter search
LR_CONFIGS = [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}]
SGD_CONFIGS = [{"alpha": 0.001}, {"alpha": 0.01}, {"alpha": 0.1}]
SPARSITY_THRESHOLD = 0.9


def _is_sparse(X):
    if hasattr(X, "toarray"):
        zero_fraction = 1.0 - (X.count_nonzero() / np.prod(X.shape))
    else:
        zero_fraction = np.mean(X == 0)
    return zero_fraction >= SPARSITY_THRESHOLD


def use_full(X):
    return X.shape[1] <= 512 or _is_sparse(X)


def find_params(X, y):
    X, y = np.asarray(X), np.asarray(y)
    do_full = use_full(X)
    cv = StratifiedShuffleSplit(n_splits=search_cv_splits(len(y)), test_size=0.2, random_state=42)

    if do_full:
        logger.info(f"Evaluating {len(LR_CONFIGS)} LogisticRegression configs...")

        def eval_config(cfg):
            scores = []
            for tr, va in cv.split(X, y):
                clf = LogisticRegression(
                    C=cfg["C"],
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=MAX_ITER_CV,
                    random_state=42,
                )
                clf.fit(X[tr], y[tr])
                scores.append(roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1]))
            return float(np.mean(scores))

        with ThreadPoolExecutor(max_workers=len(LR_CONFIGS)) as ex:
            results = list(ex.map(eval_config, LR_CONFIGS))

        best_idx = int(np.argmax(results))
        best_C = LR_CONFIGS[best_idx]["C"]
        cv_score = results[best_idx]
        logger.info(f"Best LogisticRegression C: {best_C} (AUC: {cv_score:.4f})")
        return {"C": best_C, "use_logreg": True, "cv_score": cv_score}

    else:
        logger.info(f"Evaluating {len(SGD_CONFIGS)} SGD logistic regression configs...")

        def eval_config(cfg):
            scores = []
            for tr, va in cv.split(X, y):
                clf = SGDClassifier(
                    loss="log_loss",
                    alpha=cfg["alpha"],
                    class_weight="balanced",
                    max_iter=MAX_ITER,
                    tol=1e-3,
                    random_state=42,
                )
                clf.fit(X[tr], y[tr])
                scores.append(roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1]))
            return float(np.mean(scores))

        with ThreadPoolExecutor(max_workers=len(SGD_CONFIGS)) as ex:
            results = list(ex.map(eval_config, SGD_CONFIGS))

        best_idx = int(np.argmax(results))
        best_alpha = SGD_CONFIGS[best_idx]["alpha"]
        cv_score = results[best_idx]
        logger.info(f"Best SGD alpha: {best_alpha} (AUC: {cv_score:.4f})")
        return {"alpha": best_alpha, "use_logreg": False, "cv_score": cv_score}


class Head(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=None, C=None, use_logreg=False, cv_score=None):
        self.alpha = alpha
        self.C = C
        self.use_logreg = use_logreg
        self.cv_score = cv_score

    def _fit(self, X, y):
        X = np.asarray(X)
        if self.use_logreg:
            logger.info("Fitting LogisticRegression head...")
            self.model = LogisticRegression(
                C=self.C or 1.0,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=MAX_ITER,
                random_state=42,
            )
        else:
            logger.info("Fitting SGD logistic regression head...")
            self.model = SGDClassifier(
                loss="log_loss",
                alpha=self.alpha,
                class_weight="balanced",
                max_iter=MAX_ITER,
                tol=1e-3,
                n_jobs=-1,
                random_state=42,
            )
        self.model.fit(X, y)
        self.input_dim = X.shape[1]
        return self

    def fit(self, X, y):
        self._fit(X, y)
        self.score = self.cv_score if self.cv_score is not None else 0.5
        logger.info(f"LR head score (from CV): {self.score:.4f}")
        return self

    def predict_proba(self, X):
        # LR/SGD log_loss decision_function = log-odds; sigmoid gives exact probability
        logits = self.model.decision_function(np.asarray(X))
        p = 1.0 / (1.0 + np.exp(-logits))
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def save(self, name: str, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        metadata = {
            "alpha": self.alpha,
            "C": self.C,
            "score": self.score,
            "input_dim": self.input_dim,
            "use_logreg": bool(self.use_logreg),
            "cv_score": self.cv_score,
        }
        with open(os.path.join(model_dir, f"{name}_metadata.json"), "w") as f:
            json.dump(metadata, f)
        joblib.dump(self.model, os.path.join(model_dir, f"{name}_model.joblib"))

    @classmethod
    def load(cls, name: str, model_dir: str):
        with open(os.path.join(model_dir, f"{name}_metadata.json")) as f:
            metadata = json.load(f)
        model = joblib.load(os.path.join(model_dir, f"{name}_model.joblib"))
        head = cls(
            alpha=metadata.get("alpha"),
            C=metadata.get("C"),
            use_logreg=bool(metadata.get("use_logreg", False)),
            cv_score=metadata.get("cv_score"),
        )
        head.model = model
        head.score = metadata["score"]
        head.input_dim = metadata["input_dim"]
        return head


def convert_to_onnx(name: str, model_dir: str) -> str:
    """
    Convert to ONNX: p = σ(w^T x + b)
    LR/SGD log_loss decision_function = log-odds, so sigmoid gives exact probability.
    """
    head = Head.load(name, model_dir)
    base = head.model
    input_dim = int(head.input_dim)

    w = np.asarray(base.coef_, dtype=np.float32).reshape(1, input_dim)
    b = np.asarray(base.intercept_, dtype=np.float32).reshape(1,)

    W_init = numpy_helper.from_array(w.T, name=f"{name}_W")
    b_init = numpy_helper.from_array(b, name=f"{name}_b")
    shape_init = numpy_helper.from_array(
        np.array([-1], dtype=np.int64), name=f"{name}_shape1d"
    )

    X = helper.make_tensor_value_info(
        f"input_{name}", TensorProto.FLOAT, ["batch_size", input_dim]
    )
    Y = helper.make_tensor_value_info(
        f"output_{name}", TensorProto.FLOAT, ["batch_size"]
    )

    gemm = helper.make_node(
        "Gemm", [f"input_{name}", f"{name}_W", f"{name}_b"], [f"{name}_z"]
    )
    sig = helper.make_node("Sigmoid", [f"{name}_z"], [f"{name}_p"])
    reshape = helper.make_node(
        "Reshape", [f"{name}_p", f"{name}_shape1d"], [f"output_{name}"]
    )

    graph = helper.make_graph(
        nodes=[gemm, sig, reshape],
        name=f"{name}",
        inputs=[X],
        outputs=[Y],
        initializer=[W_init, b_init, shape_init],
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
