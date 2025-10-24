import os
import numpy as np
import optuna
import json
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import onnx
from onnx import helper, numpy_helper, TensorProto

from ...utils.logging import logger
from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


MIN_FEATURES = 4
MAX_FEATURES = 2048

MAX_NUM_TRIALS = 10

N_TREES = 100


def find_params(X, y, num_trials):
    num_trials = min(num_trials, MAX_NUM_TRIALS)

    options = ["mean", "median", "1.25*mean", "1.5*mean", "2*mean"]

    num_trials = min(num_trials, len(options))

    logger.info("Starting hyperparameter optimization for SelectFromModel threshold.")

    def objective(trial):
        threshold = trial.suggest_categorical("threshold", options)

        model = RandomForestClassifier(
            n_estimators=N_TREES, random_state=42, n_jobs=-1, class_weight="balanced"
        )

        selector = SelectFromModel(model, threshold=threshold, prefit=False)

        pipe = Pipeline(
            [
                ("select", selector),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=N_TREES,
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials, show_progress_bar=True)

    best_params = study.best_params
    logger.info(f"Best threshold found: {best_params['threshold']}")

    return best_params


N_TREES = 100


class FeatureSelector:
    def __init__(self, threshold: str = None):
        self.threshold = threshold

    def fit(self, X, y):
        if self.threshold is None:
            self.selected_idx_ = np.arange(X.shape[1])
            logger.info("No feature selection performed, using all features.")
            return self

        model = RandomForestClassifier(
            n_estimators=N_TREES, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        selector = SelectFromModel(model, threshold=self.threshold, prefit=False)
        selector.fit(X, y)

        mask = selector.get_support()
        self.selected_idx_ = np.where(mask)[0].tolist()
        self.input_dim = X.shape[1]
        self.output_dim = len(self.selected_idx_)
        logger.info(f"Selected {self.output_dim}/{self.input_dim} features.")
        return self

    def transform(self, X):
        if not hasattr(self, "selected_idx_"):
            raise ValueError("Selector not fitted yet.")
        return X[:, self.selected_idx_]

    def save(self, name: str, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        metadata = {
            "threshold": self.threshold,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "selected_idx": self.selected_idx_,
        }
        with open(os.path.join(model_dir, f"{name}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(
            f"Saved selected feature indices to {model_dir}/{name}_metadata.json"
        )

    @classmethod
    def load(cls, name: str, model_dir: str):
        path = os.path.join(model_dir, f"{name}_metadata.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        with open(path, "r") as f:
            metadata = json.load(f)

        obj = cls(threshold=metadata.get("threshold"))
        obj.input_dim = metadata.get("input_dim")
        obj.output_dim = metadata.get("output_dim")
        obj.selected_idx_ = metadata.get("selected_idx", [])
        logger.info(
            f"Loaded feature selector: {len(obj.selected_idx_)} selected features."
        )
        return obj


def convert_to_onnx(name: str, model_dir: str):
    meta_path = os.path.join(model_dir, f"{name}_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    selected_idx = metadata.get("selected_idx", [])
    input_dim = int(metadata.get("input_dim", 0))
    output_dim = int(metadata.get("output_dim", len(selected_idx)))

    if not selected_idx:
        logger.info("No feature selection performed; skipping ONNX export.")
        return None

    logger.info(
        f"Converting feature selector with {output_dim}/{input_dim} features to ONNX..."
    )

    # Create ONNX constants and nodes
    indices_init = numpy_helper.from_array(
        np.array(selected_idx, dtype=np.int64), name=f"{name}_indices"
    )

    X = helper.make_tensor_value_info(
        f"input_{name}", TensorProto.FLOAT, ["batch_size", input_dim]
    )
    Y = helper.make_tensor_value_info(
        f"output_{name}", TensorProto.FLOAT, ["batch_size", output_dim]
    )

    node = helper.make_node(
        "Gather",
        inputs=[f"input_{name}", f"{name}_indices"],
        outputs=[f"output_{name}"],
        name=f"{name}_feature_selector",
        axis=1,
    )

    graph = helper.make_graph(
        nodes=[node],
        name=f"{name}",
        inputs=[X],
        outputs=[Y],
        initializer=[indices_init],
    )

    model = helper.make_model(
        graph,
        producer_name="ModelFeatureSelector",
        opset_imports=[helper.make_operatorsetid("", ONNX_TARGET_OPSET)],
    )
    model.ir_version = ONNX_IR_VERSION

    onnx.checker.check_model(model)

    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    onnx.save(model, onnx_path)
    logger.info(f"Feature selector exported to ONNX using Gather: {onnx_path}")

    return onnx_path
