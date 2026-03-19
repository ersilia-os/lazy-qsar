import json
import joblib
import os
import tempfile
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import onnx
from onnx import helper, numpy_helper, TensorProto

from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION
from ...utils.logging import logger

try:
    from zsxgboost import ZeroShotXGBClassifier
    _ZSX_AVAILABLE = True
except ImportError:
    _ZSX_AVAILABLE = False


def _require_zsxgboost():
    if not _ZSX_AVAILABLE:
        raise ImportError(
            "zsxgboost is not installed. "
            "Install it with: pip install lazyqsar[boosting]"
        )


def find_params(X, y):
    _require_zsxgboost()
    # ZeroShotXGBClassifier selects its own hyperparameters — no grid search needed.
    # cv_score is set to None; the assembler will use OOF AUC for ensemble selection.
    logger.info("ZeroShotXGBClassifier: skipping find_params CV (zero-shot auto-tuning).")
    return {"portfolio": True, "cv_score": None}


class Head(BaseEstimator, ClassifierMixin):
    def __init__(self, portfolio=True, cv_score=None):
        self.portfolio = portfolio
        self.cv_score = cv_score

    def _fit(self, X, y):
        _require_zsxgboost()
        logger.info("Fitting ZeroShotXGBClassifier head...")
        self.model = ZeroShotXGBClassifier(
            portfolio=self.portfolio, device="cpu", verbose=False, nthread=1
        )
        self.model.fit(X, y)
        self.input_dim = X.shape[1]
        return self

    def fit(self, X, y):
        self._fit(X, y)
        self.score = self.cv_score if self.cv_score is not None else 0.5
        logger.info("XGB head fitted.")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def save(self, name: str, model_dir: str):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        metadata = {
            "score": self.score,
            "input_dim": self.input_dim,
            "portfolio": self.portfolio,
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
            portfolio=metadata.get("portfolio", True),
            cv_score=metadata.get("cv_score"),
        )
        head.model = model
        head.score = metadata["score"]
        head.input_dim = metadata["input_dim"]
        return head


def convert_to_onnx(name: str, model_dir: str) -> str:
    """
    Convert ZeroShotXGBClassifier to ONNX.
    Uses clf.to_onnx() then grafts on:
      - Identity alias: input_{name} -> float_input
      - Gather(axis=1, idx=1): probabilities -> {name}_prob_pos
      - Reshape(-1): {name}_prob_pos -> output_{name}
    Output: flat 1D vector of positive-class probabilities [batch_size].
    """
    _require_zsxgboost()
    head = Head.load(name, model_dir)
    input_dim = int(head.input_dim)

    # Write zsxgboost's ONNX to a temp file, then load it
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        raw_path = f.name
    try:
        head.model.to_onnx(raw_path)
        xgb_model = onnx.load(raw_path)
    finally:
        os.remove(raw_path)

    # zsxgboost emits: input 'float_input', outputs ['label', 'probabilities']
    orig_input_name = xgb_model.graph.input[0].name
    # Find the float probabilities output (shape: [batch, 2])
    prob_output_name = None
    for out in xgb_model.graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT:
            prob_output_name = out.name
            break
    if prob_output_name is None:
        prob_output_name = xgb_model.graph.output[-1].name
    logger.info(f"XGB ONNX: orig_input={orig_input_name!r}, prob_output={prob_output_name!r}")

    # Initializers for Gather index and Reshape shape
    gather_idx = helper.make_tensor(
        f"{name}_gather_idx", TensorProto.INT64, dims=[1],
        vals=np.array([1], dtype=np.int64),
    )
    xgb_model.graph.initializer.append(gather_idx)

    flatten_shape = numpy_helper.from_array(
        np.array([-1], dtype=np.int64), name=f"{name}_flatten_shape"
    )
    xgb_model.graph.initializer.append(flatten_shape)

    # Identity alias: input_{name} -> original graph input name
    input_alias = helper.make_node(
        "Identity", inputs=[f"input_{name}"], outputs=[orig_input_name],
        name=f"{name}_InputAlias",
    )
    # Gather positive-class column
    gather_node = helper.make_node(
        "Gather",
        inputs=[prob_output_name, f"{name}_gather_idx"],
        outputs=[f"{name}_prob_pos"],
        axis=1, name=f"{name}_GatherPos",
    )
    # Reshape to 1D
    reshape_node = helper.make_node(
        "Reshape",
        inputs=[f"{name}_prob_pos", f"{name}_flatten_shape"],
        outputs=[f"output_{name}"],
        name=f"{name}_Flatten",
    )

    # Replace graph inputs/outputs
    del xgb_model.graph.input[:]
    xgb_model.graph.input.append(
        helper.make_tensor_value_info(
            f"input_{name}", TensorProto.FLOAT, ["batch_size", input_dim]
        )
    )
    del xgb_model.graph.output[:]
    xgb_model.graph.output.append(
        helper.make_tensor_value_info(
            f"output_{name}", TensorProto.FLOAT, ["batch_size"]
        )
    )

    xgb_model.graph.node.insert(0, input_alias)
    xgb_model.graph.node.extend([gather_node, reshape_node])
    xgb_model.graph.name = name

    # Ensure standard ai.onnx opset is registered (for Gather, Reshape, Identity)
    has_standard_opset = any(op.domain == "" for op in xgb_model.opset_import)
    if not has_standard_opset:
        xgb_model.opset_import.append(helper.make_opsetid("", ONNX_TARGET_OPSET))

    xgb_model.ir_version = ONNX_IR_VERSION

    onnx.checker.check_model(xgb_model)
    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    onnx.save(xgb_model, onnx_path)
    logger.info(f"XGB ONNX saved to {onnx_path}")
    return onnx_path
