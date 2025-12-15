import os
import json
import joblib
import numpy as np
import optuna
import time
from lazyqsar.utils._install_extras import ensure_torch_cpu

try:
    import torch
except ImportError:
    ensure_torch_cpu()
    import torch

import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import onnx
from onnx import helper, numpy_helper, TensorProto

from ...utils.logging import logger

from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION


MAX_NUM_TRIALS = 50

NUM_EPOCHS = 30
BATCH_SIZE = 32


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


def find_params(X, y, num_trials):
    """
    Run Optuna hyperparameter optimization for HeadNN.
    Evaluation metric: ROC AUC.
    """
    n_trials = min(num_trials, MAX_NUM_TRIALS)
    epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE

    def objective(trial):
        input_dim = X.shape[1]
        n_hidden = trial.suggest_int("n_hidden", 0, 2)
        scale1 = trial.suggest_float("scale1", 0.1, 0.5)
        scale2 = trial.suggest_float("scale2", 0.1, 0.5)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y
        )

        model = HeadNN(input_dim, n_hidden, scale1, scale2, dropout)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        pos_weight = torch.tensor(
            [(len(y_train) - y_train.sum()) / y_train.sum()], dtype=torch.float32
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)

        for epoch in range(epochs):
            model.train()
            for i in range(0, len(X_train_t), batch_size):
                xb = X_train_t[i : i + batch_size]
                yb = y_train_t[i : i + batch_size]
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t).cpu().numpy()
            preds_val = 1 / (1 + np.exp(-logits_val))
        auc = roc_auc_score(y_val, preds_val)
        return auc

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(
        {"n_hidden": 1, "scale1": 0.5, "scale2": 0.5, "dropout": 0.2, "lr": 1e-3}
    )
    study.optimize(objective, n_trials=n_trials)

    results = {
        "n_hidden": study.best_params["n_hidden"],
        "scale1": study.best_params["scale1"],
        "scale2": study.best_params["scale2"],
        "dropout": study.best_params["dropout"],
        "lr": study.best_params["lr"],
        "epochs": epochs,
        "batch_size": batch_size,
        "input_dim": X.shape[1],
    }

    return results


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
        self.model = None

    def _fit(self, X, y):
        logger.info("Fitting the MLP model...")
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

    def _calibrate(self, X, y):
        logger.info("Calibrating the model using Platt scaling...")
        splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        y_hat = []
        y_true = []
        t0 = time.time()
        for train_idx, val_idx in splitter.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            self._fit(X_train, y_train)
            logits_val = self.predict_raw(X_val)
            y_hat += list(logits_val)
            y_true += list(y_val)
            t1 = time.time()
            if (t1 - t0) > 60:
                break
        y_hat = np.array(y_hat)
        logger.debug("Shape of y_hat: {}".format(y_hat.shape))
        y_true = np.array(y_true)
        logger.debug("Shape of y_true: {}".format(y_true.shape))
        self.calibrator = LogisticRegression(class_weight="balanced")
        self.calibrator.fit(y_hat.reshape(-1, 1), y_true)
        self.score = roc_auc_score(
            y_true, self.calibrator.predict_proba(y_hat.reshape(-1, 1))[:, 1]
        )
        logger.debug("Done with calibration! Score: {:.4f}".format(self.score))
        return self.score

    def fit(self, X, y):
        self._calibrate(X, y)
        self._fit(X, y)
        return self

    def predict_raw(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t).cpu().numpy()
        return logits

    def predict_proba(self, X):
        logits = self.predict_raw(X)
        probs = self.calibrator.predict_proba(logits.reshape(-1, 1))
        return probs

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
        }
        meta_path = os.path.join(model_dir, f"{name}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        joblib.dump(
            self.calibrator, os.path.join(model_dir, f"{name}_calibrator.joblib")
        )

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
            input_dim, n_hidden, scale1, scale2, dropout, lr, epochs, batch_size, device
        )
        obj.model = model

        obj.calibrator = joblib.load(
            os.path.join(model_dir, f"{name}_calibrator.joblib")
        )
        obj.score = metadata.get("score", None)

        return obj


def convert_to_onnx(name: str, model_dir: str):
    """
    Export Torch head -> logits and append Platt calibrator in ONNX.
    Final output: flat 1D vector [batch_size].
    """
    head = Head.load(name, model_dir)
    model = head.model
    model = model.to("cpu")
    model.eval()

    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    dummy_input = torch.randn(1, head.input_dim, dtype=torch.float32, device="cpu")

    # 1) Export Torch model (produces logits_{name})
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

    # 2) Patch ONNX: add Platt = Mul + Add + Sigmoid (+ Reshape to 1D)
    onnx_model = onnx.load(onnx_path)

    coef = np.asarray(head.calibrator.coef_, dtype=np.float32).reshape(
        1,
    )  # scalar
    intercept = np.asarray(head.calibrator.intercept_, dtype=np.float32).reshape(
        1,
    )
    shape1d = np.array([-1], dtype=np.int64)

    onnx_model.graph.initializer.extend(
        [
            numpy_helper.from_array(coef, name=f"calib_coef_{name}"),
            numpy_helper.from_array(intercept, name=f"calib_intercept_{name}"),
            numpy_helper.from_array(shape1d, name=f"shape1d_{name}"),
        ]
    )

    mul_node = helper.make_node(
        "Mul",
        inputs=[f"logits_{name}", f"calib_coef_{name}"],
        outputs=[f"logits_scaled_{name}"],
        name=f"Calib_Mul_{name}",
    )
    add_node = helper.make_node(
        "Add",
        inputs=[f"logits_scaled_{name}", f"calib_intercept_{name}"],
        outputs=[f"logits_shifted_{name}"],
        name=f"Calib_Add_{name}",
    )
    sigmoid_node = helper.make_node(
        "Sigmoid",
        inputs=[f"logits_shifted_{name}"],
        outputs=[f"probs_2d_{name}"],
        name=f"Calib_Sigmoid_{name}",
    )
    reshape_node = helper.make_node(
        "Reshape",
        inputs=[f"probs_2d_{name}", f"shape1d_{name}"],
        outputs=[f"output_{name}"],
        name=f"Output_Reshape1D_{name}",
    )

    onnx_model.graph.node.extend([mul_node, add_node, sigmoid_node, reshape_node])

    # Replace graph outputs with our final 1D tensor
    del onnx_model.graph.output[:]
    onnx_model.graph.output.extend(
        [
            helper.make_tensor_value_info(
                f"output_{name}", TensorProto.FLOAT, ["batch_size"]
            )
        ]
    )

    onnx_model.graph.name = f"{name}"
    onnx_model.ir_version = ONNX_IR_VERSION

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_path)
    logger.info(f"ONNX model with calibrator saved to {onnx_path}")
    return onnx_path
