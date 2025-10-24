import json
import joblib
import os
import time
import numpy as np
import optuna
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

import onnx
from onnx import helper, numpy_helper, TensorProto

from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION
from ...utils.logging import logger


MAX_NUM_TRIALS = 100
MIN_NUM_TRIALS = 10
MAX_ITER = 1000
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


def find_params(X, y, num_trials):
    """
    Tune either:
      - alpha (= 1/C) for SGDClassifier (dense inputs)
      - C for LinearSVC (sparse inputs)
    using out-of-fold ROC-AUC on decision_function.
    """
    num_trials = min(num_trials, MAX_NUM_TRIALS)
    X = np.asarray(X)
    y = np.asarray(y)

    do_full = use_full(X)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    if do_full:
        logger.info(f"Running Optuna for LinearSVC with {num_trials} trials...")

        def objective(trial):
            C = trial.suggest_float("C", 1e-4, 1e2, log=True)
            oof = np.full(len(y), np.nan, dtype=np.float32)
            for tr, va in cv.split(X, y):
                clf = LinearSVC(
                    C=C,
                    class_weight="balanced",
                    max_iter=MAX_ITER,
                    random_state=42,
                )
                clf.fit(X[tr], y[tr])
                oof[va] = clf.decision_function(X[va]).astype(np.float32)
            if np.isnan(oof).any():
                return 0.5
            return roc_auc_score(y, oof)

        study = optuna.create_study(
            direction="maximize", pruner=optuna.pruners.MedianPruner()
        )
        study.enqueue_trial({"C": 1.0})
        study.optimize(objective, n_trials=num_trials, n_jobs=-1)
        best_C = float(study.best_params["C"])
        logger.info(f"Best LinearSVC C: {best_C}")
        return {"C": best_C, "use_linearsvc": True}

    else:
        num_trials = max(MIN_NUM_TRIALS, num_trials)

        logger.info(
            f"Dense input detected. Running Optuna for SGD hinge head with {num_trials} trials..."
        )

        def objective(trial):
            C = trial.suggest_float("C", 1e-4, 1e2, log=True)
            alpha = 1.0 / C
            oof = np.full(len(y), np.nan, dtype=np.float32)
            for tr, va in cv.split(X, y):
                clf = SGDClassifier(
                    loss="hinge",
                    alpha=alpha,
                    class_weight="balanced",
                    max_iter=MAX_ITER,
                    n_jobs=-1,
                    random_state=42,
                )
                clf.fit(X[tr], y[tr])
                oof[va] = clf.decision_function(X[va]).astype(np.float32)
            if np.isnan(oof).any():
                return 0.5
            return roc_auc_score(y, oof)

        study = optuna.create_study(
            direction="maximize", pruner=optuna.pruners.MedianPruner()
        )
        study.enqueue_trial({"C": 1.0})
        study.optimize(objective, n_trials=num_trials, n_jobs=-1)
        best_C = float(study.best_params["C"])
        best_alpha = 1.0 / best_C
        logger.info(f"Best SGD C: {best_C} (alpha={best_alpha})")
        return {"alpha": best_alpha, "use_linearsvc": False}


class Head(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=None, C=None, use_linearsvc=False):
        self.alpha = alpha
        self.C = C
        self.use_linearsvc = use_linearsvc

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
        logger.info("Calibrating decision function with LogisticRegression...")

        X = np.asarray(X)
        splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        t0 = time.time()
        y_hat = []
        y_cal = []

        for train_idxs, test_idxs in splitter.split(X, y):
            X_train, y_train = X[train_idxs], y[train_idxs]
            X_test, y_test = X[test_idxs], y[test_idxs]
            self._fit(X_train, y_train)
            t1 = time.time()
            y_hat += list(self.model.decision_function(X_test).astype(np.float32))
            y_cal += list(y_test)
            if (t1 - t0) > 60:
                break

        y_hat = np.array(y_hat)
        y_cal = np.array(y_cal)
        self.calibrator = LogisticRegression(class_weight="balanced", solver="lbfgs")
        self.calibrator.fit(y_hat.reshape(-1, 1), y_cal)
        self.score = roc_auc_score(y_cal, y_hat)
        logger.info(f"ROC-AUC (pre-calibration): {self.score:.4f}")
        return self.score

    def fit(self, X, y):
        self._calibrate(X, y)
        self._fit(X, y)
        return self

    def predict_proba(self, X):
        y_hat = self.model.decision_function(X).astype(np.float32)
        y_hat = self.calibrator.predict_proba(y_hat.reshape(-1, 1))[:, 1]
        return np.vstack([1 - y_hat, y_hat]).T

    def predict(self, X):
        y_hat = self.model.decision_function(X).astype(np.float32)
        return (self.calibrator.predict_proba(y_hat.reshape(-1, 1))[:, 1] > 0.5).astype(
            int
        )

    def save(self, name, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        metadata = {
            "alpha": self.alpha,
            "C": self.C,
            "score": self.score,
            "input_dim": self.input_dim,
            "use_linearsvc": bool(self.use_linearsvc),
        }
        with open(os.path.join(model_dir, f"{name}_metadata.json"), "w") as f:
            json.dump(metadata, f)
        joblib.dump(self.model, os.path.join(model_dir, f"{name}_model.joblib"))
        joblib.dump(
            self.calibrator, os.path.join(model_dir, f"{name}_calibrator.joblib")
        )

    @classmethod
    def load(cls, name, model_dir):
        with open(os.path.join(model_dir, f"{name}_metadata.json"), "r") as f:
            metadata = json.load(f)
        model = joblib.load(os.path.join(model_dir, f"{name}_model.joblib"))
        calibrator = joblib.load(os.path.join(model_dir, f"{name}_calibrator.joblib"))
        head = cls(
            alpha=metadata.get("alpha"),
            C=metadata.get("C"),
            use_linearsvc=metadata.get("use_linearsvc", False),
        )
        head.model = model
        head.calibrator = calibrator
        head.score = metadata["score"]
        head.input_dim = metadata["input_dim"]
        return head


def convert_to_onnx(name: str, model_dir: str) -> str:
    """
    Convert LinearSVC/SGD + calibrator into a single ONNX graph:
        p = sigmoid( a * (w^T x + b) + c )
    where (w, b) come from the linear model and (a, c) from the calibrator.
    """
    head = Head.load(name, model_dir)
    clf = head.model
    cal = head.calibrator
    input_dim = int(head.input_dim)

    # extract weights and bias
    w = np.asarray(clf.coef_, dtype=np.float32).reshape(1, input_dim)
    b = np.asarray(clf.intercept_, dtype=np.float32).reshape(
        1,
    )

    # extract calibrator parameters
    a = float(np.asarray(cal.coef_, dtype=np.float32).reshape(1, 1)[0, 0])
    c = float(
        np.asarray(cal.intercept_, dtype=np.float32).reshape(
            1,
        )[0]
    )

    # collapse affine transformation
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
    shape1d_init = numpy_helper.from_array(
        np.array([-1], np.int64), name=f"shape_out_{name}"
    )

    gemm = helper.make_node(
        "Gemm",
        inputs=[f"input_{name}", f"W2_{name}", f"b2_{name}"],
        outputs=[f"affine_out_{name}"],
        name=f"{name}_Linear_Gemm",
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
