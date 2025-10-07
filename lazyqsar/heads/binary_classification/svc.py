import json
import joblib
import os
import numpy as np
import optuna
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

import onnx
from onnx import helper, numpy_helper, TensorProto

from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION

from ...utils.logging import logger

MAX_NUM_TRIALS = 100
MAX_ITER = 1000


def find_params(X, y, num_trials):
    """
    Tune C for LinearSVC with Optuna using out-of-fold ROC-AUC on decision_function.

    Returns
    -------
    dict: {"C": best_C}
    """
    logger.info("Finding best C for SVC head...")
    X = np.asarray(X)
    y = np.asarray(y)
    kf = StratifiedKFold(n_splits=5, shuffle=True)

    if X.shape[0] > X.shape[1]:
        dual = False
    else:
        dual = True

    if dual:
        loss = "hinge"
    else:
        loss = "squared_hinge"

    n_trials = min(num_trials, MAX_NUM_TRIALS)

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)

        oof = np.full(len(y), np.nan, dtype=np.float32)
        for tr, va in kf.split(X, y):
            clf = LinearSVC(C=C, dual=dual, loss=loss, class_weight="balanced", max_iter=MAX_ITER)
            clf.fit(X[tr], y[tr])
            oof[va] = clf.decision_function(X[va]).astype(np.float32)

        if np.isnan(oof).any():
            return 0.5

        return roc_auc_score(y, oof)

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.enqueue_trial({"C": 1.0})
    study.optimize(objective, n_trials=n_trials)

    return {"C": float(study.best_params["C"]), "dual": dual}


class Head(BaseEstimator, ClassifierMixin):
    def __init__(self, C, dual):
        self.C = C
        self.dual = dual

    def fit(self, X, y):
        logger.info("Fitting SVC head...")
        if self.dual:
            loss = "hinge"
        else:
            loss = "squared_hinge"
        self.model = LinearSVC(
            C=self.C, class_weight="balanced", loss=loss, dual=self.dual, max_iter=MAX_ITER
        )
        self.model.fit(X, y)
        self.calibrate(X, y)
        self.input_dim = X.shape[1]
        return self

    def calibrate(self, X, y):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_hat = []
        y_true = []
        for train_idx, test_idx in splitter.split(X, y):
            self.model.fit(X[train_idx], y[train_idx])
            y_hat_fold = self.model.decision_function(X[test_idx]).astype(np.float32)
            y_hat += list(y_hat_fold)
            y_true += list(y[test_idx])
        self.calibrator = LogisticRegression().fit(
            np.array(y_hat).reshape(-1, 1), np.array(y_true)
        )
        self.score = roc_auc_score(y_true, y_hat)

    def predict_proba(self, X):
        y_hat = self.model.decision_function(X).astype(np.float32)
        y_hat = self.calibrator.predict_proba(y_hat.reshape(-1, 1))[:, 1]
        return np.vstack([1 - y_hat, y_hat]).T

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, name: str, model_dir: str):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        metadata = {
            "C": self.C,
            "dual": bool(self.dual),
            "score": self.score,
            "input_dim": self.input_dim,
        }
        with open(os.path.join(model_dir, f"{name}_metadata.json"), "w") as f:
            json.dump(metadata, f)
        joblib.dump(self.model, os.path.join(model_dir, f"{name}_model.joblib"))
        joblib.dump(
            self.calibrator, os.path.join(model_dir, f"{name}_calibrator.joblib")
        )

    @classmethod
    def load(cls, name: str, model_dir: str):
        with open(os.path.join(model_dir, f"{name}_metadata.json"), "r") as f:
            metadata = json.load(f)
        model = joblib.load(os.path.join(model_dir, f"{name}_model.joblib"))
        calibrator = joblib.load(os.path.join(model_dir, f"{name}_calibrator.joblib"))
        head = cls(C=metadata["C"], dual=bool(metadata["dual"]))
        head.model = model
        head.calibrator = calibrator
        head.score = metadata["score"]
        head.input_dim = metadata["input_dim"]
        return head


def convert_to_onnx(name: str, model_dir: str):
    """
    Build an ONNX graph implementing:
        p = sigmoid( a * (w^T x + b) + c )
    where (w, b) come from LinearSVC and (a, c) from the Platt calibrator (LogisticRegression).
    Collapses to a single affine + sigmoid:
        p = sigmoid( (a*w)^T x + (a*b + c) )
    Outputs a 1D vector [batch_size] of calibrated probabilities.
    """
    # ---- Load trained head ----
    head = Head.load(name, model_dir)
    svc = head.model
    cal = head.calibrator
    input_dim = int(head.input_dim)

    # ---- Extract parameters and collapse them ----
    # Linear SVC decision: z = w^T x + b
    w = np.asarray(svc.coef_, dtype=np.float32).reshape(1, input_dim)  # (1, F)
    b = np.asarray(svc.intercept_, dtype=np.float32).reshape(
        1,
    )  # (1,)

    # Platt: p = sigmoid(a * z + c)
    a = float(np.asarray(cal.coef_, dtype=np.float32).reshape(1, 1)[0, 0])  # scalar
    c = float(
        np.asarray(cal.intercept_, dtype=np.float32).reshape(
            1,
        )[0]
    )  # scalar

    # Collapse to single affine
    W2 = (a * w).T.astype(np.float32)  # (F, 1) for Gemm
    b2 = np.array([a * b[0] + c], dtype=np.float32)  # (1,)

    # ---- Build ONNX graph ----
    X = helper.make_tensor_value_info(
        f"input_{name}", TensorProto.FLOAT, ["batch_size", input_dim]
    )
    Y = helper.make_tensor_value_info(
        f"output_{name}", TensorProto.FLOAT, ["batch_size"]
    )  # 1D output

    W_init = numpy_helper.from_array(W2, name=f"W2_{name}")  # (F,1)
    b_init = numpy_helper.from_array(b2, name=f"b2_{name}")  # (1,)
    shape1d_init = numpy_helper.from_array(
        np.array([-1], np.int64), name=f"shape_out_{name}"
    )

    # Gemm: (N,F) @ (F,1) + (1,) -> (N,1)
    gemm = helper.make_node(
        "Gemm",
        inputs=[f"input_{name}", f"W2_{name}", f"b2_{name}"],
        outputs=[f"affine_out_{name}"],
        name=f"{name}_LinearSVC_Gemm",
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0,
    )
    sigm = helper.make_node(
        "Sigmoid",
        inputs=[f"affine_out_{name}"],
        outputs=[f"probs_2d_{name}"],
        name=f"{name}_Sigmoid",
    )
    # (N,1) -> (N,)
    reshape = helper.make_node(
        "Reshape",
        inputs=[f"probs_2d_{name}", f"shape_out_{name}"],
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

    opset_version = ONNX_TARGET_OPSET
    model = helper.make_model(
        graph,
        producer_name=f"{name}",
        opset_imports=[helper.make_operatorsetid("", opset_version)],
    )
    model.ir_version = ONNX_IR_VERSION

    onnx.checker.check_model(model)
    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    onnx.save(model, onnx_path)
    return onnx_path
