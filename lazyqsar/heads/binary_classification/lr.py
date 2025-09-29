import json
import joblib
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

from ...utils.logging import logger

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import onnx
from onnx import helper, numpy_helper, TensorProto

from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION

N_TRIALS = 10 # TODO increase for better tuning


def find_params(X, y):
    """
    Tune C for LogisticRegression with Optuna using out-of-fold ROC-AUC.
    Returns {"C": best_C}.
    """

    n_splits = 5
    random_state = 42
    max_iter = 1000
    n_trials = N_TRIALS

    logger.info("Finding best C for logistic regression head with Optuna...")
    X = np.asarray(X)
    y = np.asarray(y)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)

        oof = np.full(len(y), np.nan, dtype=np.float32)
        for tr, va in cv.split(X, y):
            clf = LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=random_state,
            )
            clf.fit(X[tr], y[tr])
            oof[va] = clf.predict_proba(X[va])[:, 1].astype(np.float32)

        if np.isnan(oof).any():
            return 0.5

        auc = roc_auc_score(y, oof)
        trial.report(auc, step=0)
        return auc

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    study.enqueue_trial({"C": 1.0})
    study.optimize(objective, n_trials=n_trials)

    best_C = float(study.best_params["C"])
    logger.info(f"Best C: {best_C}")
    return {"C": best_C}



class Head(BaseEstimator, ClassifierMixin):

    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        logger.info("Fitting logistic regression head...")
        self.model = LogisticRegression(C=self.C, class_weight="balanced")
        self.model.fit(X, y)
        self.calibrate(X, y)
        self.input_dim = X.shape[1]
        return self

    def calibrate(self, X, y):
        logger.info("Evaluating logistic regression head...")
        splitter = StratifiedKFold(n_splits=5, shuffle=True)
        y_hat = []
        y_true = []
        for train_idx, test_idx in splitter.split(X, y):
            self.model.fit(X[train_idx], y[train_idx])
            y_hat_fold = self.model.predict_proba(X[test_idx])[:, 1]
            y_hat += list(y_hat_fold)
            y_true += list(y[test_idx])
        self.calibrator = LogisticRegression(class_weight="balanced")
        self.calibrator.fit(np.array(y_hat).reshape(-1, 1), y_true)
        self.score = roc_auc_score(y_true, y_hat)
        logger.info(f"ROC-AUC: {self.score}")
        return self.score

    def predict_proba(self, X):
        y_hat = self.model.predict_proba(X)[:, 1]
        y_hat = self.calibrator.predict_proba(y_hat.reshape(-1, 1))[:, 1]
        return np.vstack([1 - y_hat, y_hat]).T

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, name: str, model_dir: str):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        metadata = {
            "C": self.C,
            "score": self.score,
            "input_dim": self.input_dim,
        }
        with open(os.path.join(model_dir, f"{name}_metadata.json"), "w") as f:
            json.dump(metadata, f)
        joblib.dump(self.model, os.path.join(model_dir, f"{name}_model.joblib"))
        joblib.dump(self.calibrator, os.path.join(model_dir, f"{name}_calibrator.joblib"))

    @classmethod
    def load(cls, name: str, model_dir: str):
        with open(os.path.join(model_dir, f"{name}_metadata.json"), "r") as f:
            metadata = json.load(f)
        model = joblib.load(os.path.join(model_dir, f"{name}_model.joblib"))
        calibrator = joblib.load(os.path.join(model_dir, f"{name}_calibrator.joblib"))
        head = cls(C=metadata["C"])
        head.model = model
        head.calibrator = calibrator
        head.score = metadata["score"]
        head.input_dim = metadata["input_dim"]
        return head
    

def convert_to_onnx(name: str, model_dir: str) -> str:
    """
    Convert the LogisticRegression head + probability calibrator (LogisticRegression on p)
    into a single ONNX graph that outputs a 1D vector of calibrated probabilities: [batch_size].
    Saves to {model_dir}/{name}.onnx and returns the path.
    """
    # ---- Load trained artifacts ----
    head = Head.load(name, model_dir)
    base = head.model            # sklearn.linear_model.LogisticRegression
    cal  = head.calibrator       # sklearn.linear_model.LogisticRegression (trained on p)
    input_dim = int(head.input_dim)

    # ---- Extract parameters ----
    # Base LR: p1 = sigmoid(w^T x + b)
    w = np.asarray(base.coef_, dtype=np.float32).reshape(1, input_dim)   # (1, F)
    b = np.asarray(base.intercept_, dtype=np.float32).reshape(1,)        # (1,)
    # Calibrator LR on probability p1: p2 = sigmoid(a * p1 + c)
    a = float(np.asarray(cal.coef_, dtype=np.float32).reshape(1, 1)[0, 0])   # scalar
    c = float(np.asarray(cal.intercept_, dtype=np.float32).reshape(1,)[0])   # scalar

    # Initializers
    W2 = w.T.astype(np.float32)                         # (F, 1) for Gemm B
    b2 = np.array([b[0]], dtype=np.float32)            # (1,)
    a_arr = np.array([a], dtype=np.float32)            # (1,) broadcast with (N,1)
    c_arr = np.array([c], dtype=np.float32)            # (1,)
    flat_shape = np.array([-1], dtype=np.int64)        # Reshape to 1D

    W_init = numpy_helper.from_array(W2, name=f"{name}_W")
    b_init = numpy_helper.from_array(b2, name=f"{name}_b")
    a_init = numpy_helper.from_array(a_arr, name=f"{name}_a")
    c_init = numpy_helper.from_array(c_arr, name=f"{name}_c")
    shape_init = numpy_helper.from_array(flat_shape, name=f"{name}_shape1d")

    # ---- I/O ----
    X = helper.make_tensor_value_info(f"input_{name}", TensorProto.FLOAT, ["batch_size", input_dim])
    Y = helper.make_tensor_value_info(f"output_{name}", TensorProto.FLOAT, ["batch_size"])  # 1D probs

    # ---- Nodes ----
    # Gemm: (N,F) @ (F,1) + (1,) -> (N,1)
    gemm = helper.make_node(
        "Gemm",
        inputs=[f"input_{name}", f"{name}_W", f"{name}_b"],
        outputs=[f"{name}_z1"],
        name=f"{name}_Gemm",
        alpha=1.0, beta=1.0, transA=0, transB=0
    )
    sig1 = helper.make_node(
        "Sigmoid",
        inputs=[f"{name}_z1"],
        outputs=[f"{name}_p1"],
        name=f"{name}_Sigmoid1",
    )
    mul = helper.make_node(
        "Mul",
        inputs=[f"{name}_p1", f"{name}_a"],
        outputs=[f"{name}_s1"],
        name=f"{name}_Calib_Mul",
    )
    add = helper.make_node(
        "Add",
        inputs=[f"{name}_s1", f"{name}_c"],
        outputs=[f"{name}_z2"],
        name=f"{name}_Calib_Add",
    )
    sig2 = helper.make_node(
        "Sigmoid",
        inputs=[f"{name}_z2"],
        outputs=[f"{name}_p2"],
        name=f"{name}_Sigmoid2",
    )
    # (N,1) -> (N,)
    reshape = helper.make_node(
        "Reshape",
        inputs=[f"{name}_p2", f"{name}_shape1d"],
        outputs=[f"output_{name}"],   # <-- match the declared graph output
        name=f"{name}_Reshape1D",
    )

    graph = helper.make_graph(
        nodes=[gemm, sig1, mul, add, sig2, reshape],
        name=f"{name}",
        inputs=[X],
        outputs=[Y],
        initializer=[W_init, b_init, a_init, c_init, shape_init],
    )

    # ---- Model ----
    opset_version = globals().get("ONNX_TARGET_OPSET", 16)
    model = helper.make_model(
        graph,
        producer_name=f"{name}",
        opset_imports=[helper.make_operatorsetid("", opset_version)],
    )
    ir_version = globals().get("ONNX_IR_VERSION", onnx.IR_VERSION)
    model.ir_version = ir_version

    # ---- Check & save ----
    onnx.checker.check_model(model)
    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    onnx.save(model, onnx_path)
    return onnx_path

