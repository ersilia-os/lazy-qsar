import json
import joblib
import os
import numpy as np
import optuna
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

import onnx
from onnx import helper, numpy_helper, TensorProto

from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION
from ...utils.logging import logger


MAX_NUM_TRIALS = 100
MIN_NUM_TRIALS = 5
MAX_ITER = 1000


def find_params(X, y, num_trials):
    """
    Tune alpha (= 1/C) for SGDClassifier using out-of-fold ROC-AUC.
    Returns {"alpha": best_alpha}.
    """
    num_trials = max(MIN_NUM_TRIALS, min(num_trials, MAX_NUM_TRIALS))

    logger.info(f"Running Optuna with {num_trials} trials for SGD logistic regression head...")
    X = np.asarray(X)
    y = np.asarray(y)

    n_splits = 5
    random_state = 42
    n_trials = min(num_trials, MAX_NUM_TRIALS)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)
        alpha = 1.0 / C

        oof = np.full(len(y), np.nan, dtype=np.float32)
        for tr, va in cv.split(X, y):
            clf = SGDClassifier(
                loss="log_loss",
                alpha=alpha,
                class_weight="balanced",
                max_iter=MAX_ITER,
                tol=1e-3,
                n_jobs=-1,
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
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    best_C = float(study.best_params["C"])
    best_alpha = 1.0 / best_C
    logger.info(f"Best C: {best_C} (alpha={best_alpha})")
    return {"alpha": best_alpha}


class Head(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
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
        self.calibrate(X, y)
        self.input_dim = X.shape[1]
        return self

    def calibrate(self, X, y):
        logger.info("Calibrating probabilities with logistic regression...")
        y_hat = self.model.predict_proba(X)[:, 1]
        self.calibrator = LogisticRegression(class_weight="balanced", solver="lbfgs")
        self.calibrator.fit(y_hat.reshape(-1, 1), y)
        self.score = roc_auc_score(y, y_hat)
        logger.info(f"ROC-AUC: {self.score:.4f}")
        return self.score

    def predict_proba(self, X):
        y_hat = self.model.predict_proba(X)[:, 1]
        y_hat = self.calibrator.predict_proba(y_hat.reshape(-1, 1))[:, 1]
        return np.vstack([1 - y_hat, y_hat]).T

    def predict(self, X):
        y_hat = self.model.predict_proba(X)[:, 1]
        return (y_hat > 0.5).astype(int)

    def save(self, name: str, model_dir: str):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        metadata = {
            "alpha": self.alpha,
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
        head = cls(alpha=metadata["alpha"])
        head.model = model
        head.calibrator = calibrator
        head.score = metadata["score"]
        head.input_dim = metadata["input_dim"]
        return head


def convert_to_onnx(name: str, model_dir: str) -> str:
    """
    Convert the SGDClassifier (logistic) + probability calibrator into a single ONNX graph:
        p = sigmoid( a * (w^T x + b) + c )
    where (w, b) come from SGDClassifier and (a, c) from the calibrator.
    """
    head = Head.load(name, model_dir)
    base = head.model
    cal = head.calibrator
    input_dim = int(head.input_dim)

    # ---- Extract parameters ----
    # Base SGD logistic regression: p1 = sigmoid(w^T x + b)
    w = np.asarray(base.coef_, dtype=np.float32).reshape(1, input_dim)  # (1, F)
    b = np.asarray(base.intercept_, dtype=np.float32).reshape(1,)        # (1,)

    # Calibrator: p2 = sigmoid(a * p1 + c)
    a = float(np.asarray(cal.coef_, dtype=np.float32).reshape(1, 1)[0, 0])
    c = float(np.asarray(cal.intercept_, dtype=np.float32).reshape(1,)[0])

    # ---- Collapse into a single affine transformation ----
    W2 = w.T.astype(np.float32)
    b2 = np.array([b[0]], dtype=np.float32)
    a_arr = np.array([a], dtype=np.float32)
    c_arr = np.array([c], dtype=np.float32)
    flat_shape = np.array([-1], dtype=np.int64)

    W_init = numpy_helper.from_array(W2, name=f"{name}_W")
    b_init = numpy_helper.from_array(b2, name=f"{name}_b")
    a_init = numpy_helper.from_array(a_arr, name=f"{name}_a")
    c_init = numpy_helper.from_array(c_arr, name=f"{name}_c")
    shape_init = numpy_helper.from_array(flat_shape, name=f"{name}_shape1d")

    # ---- I/O ----
    X = helper.make_tensor_value_info(f"input_{name}", TensorProto.FLOAT, ["batch_size", input_dim])
    Y = helper.make_tensor_value_info(f"output_{name}", TensorProto.FLOAT, ["batch_size"])

    # ---- Nodes ----
    gemm = helper.make_node(
        "Gemm",
        inputs=[f"input_{name}", f"{name}_W", f"{name}_b"],
        outputs=[f"{name}_z1"],
        name=f"{name}_Gemm",
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0,
    )
    sig1 = helper.make_node(
        "Sigmoid", inputs=[f"{name}_z1"], outputs=[f"{name}_p1"], name=f"{name}_Sigmoid1"
    )
    mul = helper.make_node(
        "Mul", inputs=[f"{name}_p1", f"{name}_a"], outputs=[f"{name}_s1"], name=f"{name}_Mul"
    )
    add = helper.make_node(
        "Add", inputs=[f"{name}_s1", f"{name}_c"], outputs=[f"{name}_z2"], name=f"{name}_Add"
    )
    sig2 = helper.make_node(
        "Sigmoid", inputs=[f"{name}_z2"], outputs=[f"{name}_p2"], name=f"{name}_Sigmoid2"
    )
    reshape = helper.make_node(
        "Reshape",
        inputs=[f"{name}_p2", f"{name}_shape1d"],
        outputs=[f"output_{name}"],
        name=f"{name}_Reshape1D",
    )

    graph = helper.make_graph(
        nodes=[gemm, sig1, mul, add, sig2, reshape],
        name=f"{name}",
        inputs=[X],
        outputs=[Y],
        initializer=[W_init, b_init, a_init, c_init, shape_init],
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
