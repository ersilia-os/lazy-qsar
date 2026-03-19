import os
import json
import joblib
import numpy as np
from lazyqsar.utils._install_extras import ensure_torch_cpu

try:
    import torch
except ImportError:
    ensure_torch_cpu()
    import torch

import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import onnx
from onnx import helper, numpy_helper, TensorProto

from ...utils.logging import logger

from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION


NUM_EPOCHS = 30
BATCH_SIZE = 32

MLP_CONFIGS = [
    {"n_hidden": 0, "scale1": 0.5, "scale2": 0.5, "dropout": 0.0, "lr": 5e-3},
    {"n_hidden": 1, "scale1": 0.5, "scale2": 0.5, "dropout": 0.2, "lr": 1e-3},
    {"n_hidden": 2, "scale1": 0.3, "scale2": 0.5, "dropout": 0.3, "lr": 5e-4},
]


class HeadNN(nn.Module):
    """
    HeadNN is a small neural network designed for binary classification tasks.
    It supports configurations with up to two hidden layers.
    """

    def __init__(self, input_dim, n_hidden, scale1, scale2, dropout):
        super().__init__()
        layers = []
        if n_hidden == 0:
            layers.append(nn.Linear(input_dim, 1))
        elif n_hidden == 1:
            h1 = max(1, int(input_dim * scale1))
            layers.extend(
                [
                    nn.Linear(input_dim, h1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(h1, 1),
                ]
            )
        else:
            h1 = max(1, int(input_dim * scale1))
            h2 = max(1, int(h1 * scale2))
            layers.extend(
                [
                    nn.Linear(input_dim, h1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(h1, h2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(h2, 1),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def find_params(X, y):
    logger.info(f"Evaluating {len(MLP_CONFIGS)} MLP configs.")
    epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE
    input_dim = X.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    pos_weight = torch.tensor(
        [(len(y_train) - y_train.sum()) / y_train.sum()], dtype=torch.float32
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    results = []
    for cfg in MLP_CONFIGS:
        torch.manual_seed(42)
        model = HeadNN(input_dim, cfg["n_hidden"], cfg["scale1"], cfg["scale2"], cfg["dropout"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        for _ in range(epochs):
            model.train()
            for i in range(0, len(X_train_t), batch_size):
                xb = X_train_t[i: i + batch_size]
                yb = y_train_t[i: i + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t).cpu().numpy()
            preds_val = 1 / (1 + np.exp(-logits_val))
        auc = roc_auc_score(y_val, preds_val)
        results.append(float(auc))
        logger.info(f"  MLP config {cfg}: AUC={auc:.4f}")

    best_idx = int(np.argmax(results))
    best_cfg = MLP_CONFIGS[best_idx]
    cv_score = results[best_idx]
    logger.info(f"Best MLP config: {best_cfg} (AUC: {cv_score:.4f})")
    return {
        **best_cfg,
        "epochs": epochs,
        "batch_size": batch_size,
        "input_dim": input_dim,
        "cv_score": cv_score,
    }


class Head(BaseEstimator, ClassifierMixin):
    """
    Binary classification head wrapping HeadNN, trained with BCEWithLogitsLoss and class weighting.
    """

    def __init__(
        self,
        input_dim,
        n_hidden=1,
        scale1=0.5,
        scale2=0.5,
        dropout=0.0,
        lr=1e-3,
        epochs=30,
        batch_size=32,
        device=None,
        cv_score=None,
    ):
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.scale1 = scale1
        self.scale2 = scale2
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cv_score = cv_score
        self.model = None

    def _fit(self, X, y):
        logger.info("Fitting the MLP model...")
        torch.manual_seed(42)
        self.model = HeadNN(
            self.input_dim, self.n_hidden, self.scale1, self.scale2, self.dropout
        ).to(self.device)

        pos_weight = torch.tensor(
            [(len(y) - y.sum()) / y.sum()], dtype=torch.float32
        ).to(self.device)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)

        for _ in range(self.epochs):
            self.model.train()
            for i in range(0, len(X_t), self.batch_size):
                xb = X_t[i : i + self.batch_size]
                yb = y_t[i : i + self.batch_size]
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
        return self

    def fit(self, X, y):
        self._fit(X, y)
        self.score = self.cv_score if self.cv_score is not None else 0.5
        logger.info(f"MLP head score (from CV): {self.score:.4f}")
        return self

    def predict_raw(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t).cpu().numpy()
        return logits

    def predict_proba(self, X):
        # BCEWithLogitsLoss trains HeadNN to output log-odds; sigmoid gives exact probability
        logits = self.predict_raw(X)
        p = 1.0 / (1.0 + np.exp(-logits))
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def save(self, name: str, model_dir: str):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f"{name}.pth")
        torch.save(self.model.state_dict(), model_path)
        metadata = {
            "input_dim": self.input_dim,
            "n_hidden": self.n_hidden,
            "scale1": self.scale1,
            "scale2": self.scale2,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": self.device,
            "score": self.score,
            "cv_score": self.cv_score,
        }
        meta_path = os.path.join(model_dir, f"{name}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, name: str, model_dir: str):
        meta_path = os.path.join(model_dir, f"{name}_metadata.json")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        input_dim = metadata["input_dim"]
        n_hidden = metadata["n_hidden"]
        scale1 = metadata["scale1"]
        scale2 = metadata["scale2"]
        dropout = metadata["dropout"]
        lr = metadata["lr"]
        epochs = metadata["epochs"]
        batch_size = metadata["batch_size"]
        device = metadata["device"]

        model_path = os.path.join(model_dir, f"{name}.pth")
        model = HeadNN(input_dim, n_hidden, scale1, scale2, dropout).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        obj = cls(
            input_dim, n_hidden, scale1, scale2, dropout, lr, epochs, batch_size, device,
            cv_score=metadata.get("cv_score"),
        )
        obj.model = model
        obj.score = metadata.get("score", None)
        return obj


def convert_to_onnx(name: str, model_dir: str):
    """
    Export Torch head -> logits, then append Sigmoid + Reshape to produce
    calibrated probabilities. BCEWithLogitsLoss trains HeadNN to output log-odds,
    so sigmoid gives the exact probability.
    Final output: flat 1D vector [batch_size].
    """
    head = Head.load(name, model_dir)
    model = head.model.to("cpu")
    model.eval()

    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    dummy_input = torch.randn(1, head.input_dim, dtype=torch.float32, device="cpu")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=[f"input_{name}"],
        output_names=[f"logits_{name}"],
        dynamic_axes={
            f"input_{name}": {0: "batch_size"},
            f"logits_{name}": {0: "batch_size"},
        },
        opset_version=ONNX_TARGET_OPSET,
    )

    onnx_model = onnx.load(onnx_path)

    shape1d = np.array([-1], dtype=np.int64)
    onnx_model.graph.initializer.append(
        numpy_helper.from_array(shape1d, name=f"shape1d_{name}")
    )

    sigmoid_node = helper.make_node(
        "Sigmoid",
        inputs=[f"logits_{name}"],
        outputs=[f"probs_{name}"],
        name=f"Sigmoid_{name}",
    )
    reshape_node = helper.make_node(
        "Reshape",
        inputs=[f"probs_{name}", f"shape1d_{name}"],
        outputs=[f"output_{name}"],
        name=f"Reshape1D_{name}",
    )

    onnx_model.graph.node.extend([sigmoid_node, reshape_node])

    del onnx_model.graph.output[:]
    onnx_model.graph.output.append(
        helper.make_tensor_value_info(f"output_{name}", TensorProto.FLOAT, ["batch_size"])
    )

    onnx_model.graph.name = f"{name}"
    onnx_model.ir_version = ONNX_IR_VERSION

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_path)
    logger.info(f"MLP ONNX saved to {onnx_path}")
    return onnx_path
