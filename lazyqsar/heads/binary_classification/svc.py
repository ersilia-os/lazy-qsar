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
    Tune alpha (1/C) for SGDClassifier using out-of-fold ROC-AUC on decision_function.

    Returns
    -------
    dict: {"alpha": best_alpha}
    """

    num_trials = max(MIN_NUM_TRIALS, min(num_trials, MAX_NUM_TRIALS))

    logger.info("Finding best alpha for SGD head...")
    X = np.asarray(X)
    y = np.asarray(y)
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    n_trials = min(num_trials, MAX_NUM_TRIALS)

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)
        alpha = 1.0 / C

        oof = np.full(len(y), np.nan, dtype=np.float32)
        for tr, va in kf.split(X, y):
            clf = SGDClassifier(
                loss="hinge",
                alpha=alpha,
                class_weight="balanced",
                max_iter=MAX_ITER,
                tol=1e-3,
                n_jobs=-1,
                random_state=42,
            )
            clf.fit(X[tr], y[tr])
            oof[va] = clf.decision_function(X[va]).astype(np.float32)

        if np.isnan(oof).any():
            return 0.5
        return roc_auc_score(y, oof)

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
        logger.info("Fitting SGD head...")
        self.model = SGDClassifier(
            loss="hinge",
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
        logger.info("Calibrating SGD head with logistic regression...")
        y_hat = self.model.decision_function(X).astype(np.float32)
        self.calibrator = LogisticRegression().fit(y_hat.reshape(-1, 1), y)
        self.score = roc_auc_score(y, y_hat)
        logger.info(f"ROC-AUC: {self.score:.4f}")
        return self.score

    def predict_proba(self, X):
        y_hat = self.model.decision_function(X).astype(np.float32)
        y_hat = self.calibrator.predict_proba(y_hat.reshape(-1, 1))[:, 1]
        return np.vstack([1 - y_hat, y_hat]).T

    def predict(self, X):
        y_hat = self.model.decision_function(X).astype(np.float32)
        return (self.calibrator.predict_proba(y_hat.reshape(-1, 1))[:, 1] > 0.5).astype(int)

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


def convert_to_onnx(name: str, model_dir: str):
    """
    Build an ONNX graph implementing:
        p = sigmoid( a * (w^T x + b) + c )
    where (w, b) come from SGDClassifier and (a, c) from the calibrator.
    Collapses to a single affine + sigmoid:
        p = sigmoid( (a*w)^T x + (a*b + c) )
    Outputs a 1D vector [batch_size] of calibrated probabilities.
    """
    head = Head.load(name, model_dir)
    clf = head.model
    cal = head.calibrator
    input_dim = int(head.input_dim)

    # SGDClassifier decision: z = w^T x + b
    w = np.asarray(clf.coef_, dtype=np.float32).reshape(1, input_dim)
    b = np.asarray(clf.intercept_, dtype=np.float32).reshape(1,)

    # Logistic calibrator: p = sigmoid(a * z + c)
    a = float(np.asarray(cal.coef_, dtype=np.float32).reshape(1, 1)[0, 0])
    c = float(np.asarray(cal.intercept_, dtype=np.float32).reshape(1,)[0])

    # Collapse affine transformation
    W2 = (a * w).T.astype(np.float32)
    b2 = np.array([a * b[0] + c], dtype=np.float32)

    X = helper.make_tensor_value_info(
        f"input_{name}", TensorProto.FLOAT, ["batch_size", input_dim]
    )
    Y = helper.make_tensor_value_info(
        f"output_{name}", TensorProto.FLOAT, ["batch_size"]
    )

    W_init = numpy_helper.from_array(W2, name=f"W2_{name}")
    b_init = numpy_helper.from_array(b2, name=f"b2_{name}")
    shape1d_init = numpy_helper.from_array(np.array([-1], np.int64), name=f"shape_out_{name}")

    gemm = helper.make_node(
        "Gemm",
        inputs=[f"input_{name}", f"W2_{name}", f"b2_{name}"],
        outputs=[f"affine_out_{name}"],
        name=f"{name}_SGD_Gemm",
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
