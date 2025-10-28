import json
import joblib
import os
import time
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnx import compose
from onnx import helper, TensorProto

from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION
from ...utils.logging import logger

MAX_NUM_TRIALS = 100
MIN_NUM_TRIALS = 10


def find_params(X, y, num_trials):
    logger.info(
        "Starting hyperparameter optimization for ExtraTreesClassifier (light mode)."
    )

    num_trials = min(num_trials, MAX_NUM_TRIALS)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 50, step=10),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, None]
            ),
        }

        model = ExtraTreesClassifier(
            **params,
            bootstrap=False,
            n_jobs=1,
            random_state=42,
            class_weight="balanced",
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")

    fixed_params = {
        "n_estimators": 30,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }

    study.enqueue_trial(fixed_params)
    study.optimize(objective, n_trials=num_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best cross-validated ROC-AUC: {best_score:.4f}")

    return best_params


class Head(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = None,
        max_depth: int = None,
        min_samples_split: int = None,
        min_samples_leaf: int = None,
        max_features=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def _fit(self, X, y):
        X = np.asarray(X)
        logger.info("Fitting ExtraTreesClassifier head...")
        self.model = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=False,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self.input_dim = X.shape[1]
        return self

    def _calibrate(self, X, y):
        logger.info("Calibrating probabilities with logistic regression...")
        X = np.asarray(X)
        splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        t0 = time.time()
        y_hat, y_cal = [], []

        for train_idxs, test_idxs in splitter.split(X, y):
            X_train, y_train = X[train_idxs], y[train_idxs]
            X_test, y_test = X[test_idxs], y[test_idxs]
            self._fit(X_train, y_train)
            y_hat += list(self.model.predict_proba(X_test)[:, 1])
            y_cal += list(y_test)
            if (time.time() - t0) > 60:
                break

        y_hat, y_cal = np.array(y_hat), np.array(y_cal)
        self.calibrator = LogisticRegression(class_weight="balanced", solver="lbfgs")
        self.calibrator.fit(y_hat.reshape(-1, 1), y_cal)
        self.score = roc_auc_score(y_cal, y_hat)
        logger.info(f"ROC-AUC: {self.score:.4f}")
        return self.score

    def fit(self, X, y):
        self._calibrate(X, y)
        self._fit(X, y)
        return self

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
            "score": self.score,
            "input_dim": self.input_dim,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
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
        head = cls(
            n_estimators=metadata.get("n_estimators", None),
            max_depth=metadata.get("max_depth", None),
            min_samples_split=metadata.get("min_samples_split", None),
            min_samples_leaf=metadata.get("min_samples_leaf", None),
            max_features=metadata.get("max_features", None),
        )
        head.model = model
        head.score = metadata["score"]
        head.input_dim = metadata["input_dim"]
        head.calibrator = joblib.load(
            os.path.join(model_dir, f"{name}_calibrator.joblib")
        )
        return head


def convert_to_onnx(name: str, model_dir: str) -> str:
    model_path = os.path.join(model_dir, f"{name}_model.joblib")
    calibrator_path = os.path.join(model_dir, f"{name}_calibrator.joblib")
    metadata_path = os.path.join(model_dir, f"{name}_metadata.json")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    base_model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)
    input_dim = int(metadata["input_dim"])

    logger.info("Converting ExtraTrees and Calibrator separately to ONNX...")

    et_onnx_path = os.path.join(model_dir, f"{name}_et.onnx")
    et_initial_type = [("input", FloatTensorType([None, input_dim]))]
    et_onnx = convert_sklearn(
        base_model,
        initial_types=et_initial_type,
        target_opset=ONNX_TARGET_OPSET,
        options={id(base_model): {"zipmap": False}},
    )

    prob_output_name = et_onnx.graph.output[-1].name
    logger.info(f"Detected probability output: {prob_output_name}")

    gather_idx = helper.make_tensor(
        f"{name}_gather_idx",
        TensorProto.INT64,
        dims=[1],
        vals=np.array([1], dtype=np.int64),
    )
    gather_node = helper.make_node(
        "Gather",
        inputs=[prob_output_name, f"{name}_gather_idx"],
        outputs=[f"{name}_prob_pos"],
        axis=1,
        name=f"{name}_Gather_PositiveClass",
    )

    et_onnx.graph.initializer.append(gather_idx)
    et_onnx.graph.node.append(gather_node)

    et_onnx.graph.output.append(
        helper.make_tensor_value_info(f"{name}_prob_pos", TensorProto.FLOAT, [None, 1])
    )

    onnx.save(et_onnx, et_onnx_path)

    calib_onnx_path = os.path.join(model_dir, f"{name}_calib.onnx")
    calib_initial_type = [("input", FloatTensorType([None, 1]))]
    calib_onnx = convert_sklearn(
        calibrator,
        initial_types=calib_initial_type,
        target_opset=ONNX_TARGET_OPSET,
        options={id(calibrator): {"zipmap": False}},
    )
    onnx.save(calib_onnx, calib_onnx_path)

    et_model = compose.add_prefix(onnx.load(et_onnx_path), f"{name}_et_")
    calib_model = compose.add_prefix(onnx.load(calib_onnx_path), f"{name}_calib_")

    et_probs_output = et_model.graph.output[-1].name

    calib_input_name = calib_model.graph.input[0].name

    merged = compose.merge_models(
        et_model, calib_model, io_map=[(et_probs_output, calib_input_name)]
    )

    logger.info("Merged ExtraTrees + Calibrator into single ONNX graph.")

    orig_output_name = merged.graph.output[-1].name
    slice_output_name = f"{name}_pos_prob"

    slice_starts = np.array([1], dtype=np.int64)
    slice_ends = np.array([2], dtype=np.int64)
    slice_axes = np.array([1], dtype=np.int64)

    merged.graph.initializer.extend(
        [
            helper.make_tensor(
                f"{name}_slice_starts", TensorProto.INT64, [1], slice_starts
            ),
            helper.make_tensor(
                f"{name}_slice_ends", TensorProto.INT64, [1], slice_ends
            ),
            helper.make_tensor(
                f"{name}_slice_axes", TensorProto.INT64, [1], slice_axes
            ),
        ]
    )

    slice_node = helper.make_node(
        "Slice",
        inputs=[
            orig_output_name,
            f"{name}_slice_starts",
            f"{name}_slice_ends",
            f"{name}_slice_axes",
        ],
        outputs=[slice_output_name],
        name=f"{name}_SlicePositiveClass",
    )
    merged.graph.node.append(slice_node)

    flatten_output = f"{name}_flat_output"
    flatten_shape_name = f"{name}_flatten_shape"
    flatten_shape = np.array([-1], dtype=np.int64)

    merged.graph.initializer.append(
        helper.make_tensor(flatten_shape_name, TensorProto.INT64, [1], flatten_shape)
    )

    flatten_node = helper.make_node(
        "Reshape",
        inputs=[slice_output_name, flatten_shape_name],
        outputs=[flatten_output],
        name=f"{name}_FlattenOutput",
    )
    merged.graph.node.append(flatten_node)

    orig_input_name = merged.graph.input[0].name
    input_alias = f"input_{name}"
    output_alias = f"output_{name}"

    batch_dim = "batch_size"

    input_node = helper.make_node(
        "Identity",
        inputs=[input_alias],
        outputs=[orig_input_name],
        name=f"InputFixer_{name}",
    )
    output_node = helper.make_node(
        "Identity",
        inputs=[flatten_output],
        outputs=[output_alias],
        name=f"OutputFixer_{name}",
    )

    del merged.graph.input[:]
    merged.graph.input.append(
        helper.make_tensor_value_info(
            input_alias, TensorProto.FLOAT, [batch_dim, input_dim]
        )
    )

    del merged.graph.output[:]
    merged.graph.output.append(
        helper.make_tensor_value_info(output_alias, TensorProto.FLOAT, [batch_dim])
    )

    merged.graph.node.insert(0, input_node)
    merged.graph.node.append(output_node)

    merged.graph.name = name
    merged.ir_version = ONNX_IR_VERSION

    onnx.checker.check_model(merged)
    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    onnx.save(merged, onnx_path)

    logger.info(f"Successfully merged ONNX model: {onnx_path}")
    logger.info(f"Inputs: {[i.name for i in merged.graph.input]}")
    logger.info(f"Outputs: {[o.name for o in merged.graph.output]}")
    logger.info(
        f"ONNX output shape: {[dim for dim in merged.graph.output[0].type.tensor_type.shape.dim]}"
    )

    return onnx_path
