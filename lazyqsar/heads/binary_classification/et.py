import json
import joblib
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnx import helper, numpy_helper, TensorProto

from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION
from ...utils.logging import logger
from . import search_cv_splits

ET_CONFIGS = [
    {"n_estimators": 30, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 50, "max_depth": 4, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "log2"},
    {"n_estimators": 40, "max_depth": 3, "min_samples_split": 3, "min_samples_leaf": 1, "max_features": 0.5},
]


def find_params(X, y):
    logger.info(f"Evaluating {len(ET_CONFIGS)} ExtraTreesClassifier configs.")
    cv = StratifiedShuffleSplit(n_splits=search_cv_splits(len(y)), test_size=0.2, random_state=42)

    def eval_config(cfg):
        model = ExtraTreesClassifier(
            **cfg,
            bootstrap=False,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
        scores = []
        for tr, va in cv.split(X, y):
            model.fit(X[tr], y[tr])
            scores.append(roc_auc_score(y[va], model.predict_proba(X[va])[:, 1]))
        return float(np.mean(scores))

    with ThreadPoolExecutor(max_workers=len(ET_CONFIGS)) as ex:
        results = list(ex.map(eval_config, ET_CONFIGS))

    best_idx = int(np.argmax(results))
    best_cfg = ET_CONFIGS[best_idx]
    cv_score = results[best_idx]
    logger.info(f"Best ET config: {best_cfg} (AUC: {cv_score:.4f})")
    return {**best_cfg, "cv_score": cv_score}


class Head(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = None,
        max_depth: int = None,
        min_samples_split: int = None,
        min_samples_leaf: int = None,
        max_features=None,
        cv_score=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.cv_score = cv_score

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

    def fit(self, X, y):
        self._fit(X, y)
        self.score = self.cv_score if self.cv_score is not None else 0.5
        logger.info(f"ET head score (from CV): {self.score:.4f}")
        return self

    def predict_proba(self, X):
        # predict_proba is already a probability in [0, 1]
        return self.model.predict_proba(X)

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
            "cv_score": self.cv_score,
        }
        with open(os.path.join(model_dir, f"{name}_metadata.json"), "w") as f:
            json.dump(metadata, f)
        joblib.dump(self.model, os.path.join(model_dir, f"{name}_model.joblib"))

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
            cv_score=metadata.get("cv_score"),
        )
        head.model = model
        head.score = metadata["score"]
        head.input_dim = metadata["input_dim"]
        return head


def convert_to_onnx(name: str, model_dir: str) -> str:
    """
    Convert ExtraTrees to ONNX. predict_proba is already a probability — no calibrator needed.
    Output: flat 1D vector of positive-class probabilities [batch_size].
    """
    head = Head.load(name, model_dir)
    input_dim = int(head.input_dim)

    # Convert ExtraTrees to ONNX
    initial_type = [("input", FloatTensorType([None, input_dim]))]
    et_onnx = convert_sklearn(
        head.model,
        initial_types=initial_type,
        target_opset=ONNX_TARGET_OPSET,
        options={id(head.model): {"zipmap": False}},
    )

    # Find the probability output (last output of skl2onnx ET model)
    prob_output_name = et_onnx.graph.output[-1].name
    logger.info(f"ET ONNX probability output: {prob_output_name}")

    # Add nodes to: rename input, gather positive class, reshape to 1D, rename output
    orig_input_name = et_onnx.graph.input[0].name

    gather_idx = helper.make_tensor(
        f"{name}_gather_idx", TensorProto.INT64, dims=[1],
        vals=np.array([1], dtype=np.int64),
    )
    et_onnx.graph.initializer.append(gather_idx)

    flatten_shape = numpy_helper.from_array(
        np.array([-1], dtype=np.int64), name=f"{name}_flatten_shape"
    )
    et_onnx.graph.initializer.append(flatten_shape)

    gather_node = helper.make_node(
        "Gather",
        inputs=[prob_output_name, f"{name}_gather_idx"],
        outputs=[f"{name}_prob_pos"],
        axis=1, name=f"{name}_GatherPos",
    )
    reshape_node = helper.make_node(
        "Reshape",
        inputs=[f"{name}_prob_pos", f"{name}_flatten_shape"],
        outputs=[f"output_{name}"],
        name=f"{name}_Flatten",
    )
    input_alias = helper.make_node(
        "Identity", inputs=[f"input_{name}"], outputs=[orig_input_name],
        name=f"{name}_InputAlias",
    )

    del et_onnx.graph.input[:]
    et_onnx.graph.input.append(
        helper.make_tensor_value_info(f"input_{name}", TensorProto.FLOAT, ["batch_size", input_dim])
    )
    del et_onnx.graph.output[:]
    et_onnx.graph.output.append(
        helper.make_tensor_value_info(f"output_{name}", TensorProto.FLOAT, ["batch_size"])
    )

    et_onnx.graph.node.insert(0, input_alias)
    et_onnx.graph.node.extend([gather_node, reshape_node])

    et_onnx.graph.name = name
    et_onnx.ir_version = ONNX_IR_VERSION

    onnx.checker.check_model(et_onnx)
    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    onnx.save(et_onnx, onnx_path)
    logger.info(f"ET ONNX saved to {onnx_path}")
    return onnx_path
