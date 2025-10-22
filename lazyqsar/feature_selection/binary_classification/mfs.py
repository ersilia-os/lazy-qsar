import os
import numpy as np
import optuna
import json
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

from ...utils.logging import logger
from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


MIN_FEATURES = 4
MAX_FEATURES = 2048

MAX_NUM_TRIALS = 10


def find_params(X, y, num_trials):

    results = {"threshold": 0.5}

    return results


class ModelFeatureSelector(object):

    def __init__(self, threshold: float = None):
        self.threshold = threshold
        
    def fit(self, X, y):
        if self.threshold is None:
            self.selector = None
            return self
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
        self.selector = SelectFromModel(model, threshold=self.threshold, prefit=False)
        self.selector.fit(X, y)
        return self

    def transform(self, X, y=None):
        if not hasattr(self, "selector"):
            raise ValueError("The model feature selector has not been fitted yet.")
        if self.selector is None:
            return X
        X = self.selector.transform(X)
        return X

    def save(self, name: str, model_dir: str):
        metadata = {
            "threshold": self.threshold,
        }
        meta_path = os.path.join(model_dir, f"{name}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        joblib_path = os.path.join(model_dir, f"{name}.joblib")
        joblib.dump(self.selector, joblib_path)

    @classmethod
    def load(cls, name: str, model_dir: str):
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist.")
        meta_path = os.path.join(model_dir, f"{name}_metadata.json")
        if not os.path.exists(meta_path):
            raise ValueError(f"Metadata file {meta_path} does not exist.")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        threshold = metadata.get("threshold", None)
        selector_path = os.path.join(model_dir, f"{name}.joblib")
        if not os.path.exists(selector_path):
            raise ValueError(f"Selector file {selector_path} does not exist.")
        selector = joblib.load(selector_path)
        obj = cls(k_features=threshold)
        obj.selector = selector
        return obj


def convert_to_onnx(name, model_dir: str):

    feature_selector = ModelFeatureSelector.load(name, model_dir)
    if feature_selector.selector is None:
        logger.info("No model feature selection was performed. Skipping ONNX conversion.")
        return None

    selector = feature_selector.selector
    initial_type = [
        (f"input_{name}", FloatTensorType([None, selector.scores_.shape[0]]))
    ]
    onnx_model = skl2onnx.convert_sklearn(
        selector, initial_types=initial_type, target_opset=ONNX_TARGET_OPSET
    )

    onnx_model.graph.name = f"{name}"
    onnx_model.ir_version = ONNX_IR_VERSION
    onnx_model.graph.input[0].name = f"input_{name}"
    onnx_model.graph.output[0].name = f"output_{name}"

    for node in onnx_model.graph.node:
        if f"_{name}" not in node.name:
            node.name = f"{node.name}_{name}"

    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info(f"Model based feature selector converted to ONNX and saved at {onnx_path}.")
    return onnx_path