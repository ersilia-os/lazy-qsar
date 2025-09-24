import os
import json
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from ..utils.logging import logger


NUM_TRIALS = 1  # TODO: increase
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
            layers.extend([
                nn.Linear(input_dim, h1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h1, 1),
            ])
        else:
            h1 = max(1, int(input_dim * scale1))
            h2 = max(1, int(h1 * scale2))
            layers.extend([
                nn.Linear(input_dim, h1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h2, 1),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def find_head_params(X, y):
    """
    Run Optuna hyperparameter optimization for HeadNN.
    Evaluation metric: ROC AUC.
    """
    n_trials = NUM_TRIALS
    epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE

    def objective(trial):
        input_dim = X.shape[1]
        n_hidden = trial.suggest_int("n_hidden", 0, 2)
        scale1 = trial.suggest_float("scale1", 0.1, 1.0)
        scale2 = trial.suggest_float("scale2", 0.1, 1.0)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y
        )

        model = HeadNN(input_dim, n_hidden, scale1, scale2, dropout)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        pos_weight = torch.tensor(
            [(len(y_train) - y_train.sum()) / y_train.sum()],
            dtype=torch.float32
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        for epoch in range(epochs):
            model.train()
            for i in range(0, len(X_train_t), batch_size):
                xb = X_train_t[i:i+batch_size]
                yb = y_train_t[i:i+batch_size]
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
    study.enqueue_trial({"n_hidden": 0, "scale1": 1, "scale2": 1, "dropout": 0, "lr": 1e-3})
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


class HeadForBinaryClassification(BaseEstimator, ClassifierMixin):
    """
    Binary classification head wrapping HeadNN, trained with BCEWithLogitsLoss and class weighting.
    """

    def __init__(self, input_dim, n_hidden=1, scale1=0.5, scale2=0.5, dropout=0.0,
                 lr=1e-3, epochs=30, batch_size=32, device=None):
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

    def fit(self, X, y):
        self.model = HeadNN(
            self.input_dim, self.n_hidden, self.scale1, self.scale2, self.dropout
        ).to(self.device)

        pos_weight = torch.tensor(
            [(len(y) - y.sum()) / y.sum()],
            dtype=torch.float32
        ).to(self.device)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            for i in range(0, len(X_t), self.batch_size):
                xb = X_t[i:i+self.batch_size]
                yb = y_t[i:i+self.batch_size]
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t).cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))  # sigmoid
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def save(self, model_dir: str):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, "head_nn.pth")
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
        }
        meta_path = os.path.join(model_dir, "head_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, model_dir: str):
        meta_path = os.path.join(model_dir, "head_metadata.json")
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
        
        model_path = os.path.join(model_dir, "head_nn.pth")
        model = HeadNN(
            input_dim, n_hidden, scale1, scale2, dropout
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        obj = cls(
            input_dim, n_hidden, scale1, scale2, dropout, lr, epochs, batch_size, device
            )
        obj.model = model

        return obj
    

from .. import ONNX_TARGET_OPSET, ONNX_IR_VERSION
import onnx

def convert_to_onnx(model_dir: str):
    """
    Convert a binary classification model to ONNX format.
    This function loads a binary classification model from the specified directory,
    converts it to the ONNX format for interoperability, and saves the ONNX model
    as 'head.onnx' in the specified directory.
    
    Parameters
    ----------
    model_dir : str
        The directory where the model is stored and where the ONNX file will be saved.

    Returns
    -------
    str
        The path to the saved ONNX model file.
    
    Notes
    -----
    - The function assumes that the model is compatible with PyTorch's ONNX export functionality.
    - The ONNX model is saved with dynamic axes for the batch size, allowing for variable batch sizes during inference.
    
    Examples
    --------
    >>> convert_to_onnx("/path/to/model_dir")
    ONNX model saved to /path/to/model_dir/head.onnx
    """
    head = HeadForBinaryClassification.load(model_dir)
    model = head.model
    model.eval()
    dummy_input = torch.randn(1, head.input_dim)
    onnx_path = os.path.join(model_dir, "head.onnx")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input_head'], 
        output_names=['output_head'],
        dynamic_axes={'input_head': {0: 'batch_size'}, 'output_head': {0: 'batch_size'}},
        opset_version=ONNX_TARGET_OPSET,
    )
    onnx_model = onnx.load(onnx_path)
    onnx_model.graph.name = "Head"
    onnx_model.ir_version = ONNX_IR_VERSION
    onnx.save(onnx_model, onnx_path)
    logger.info(f"ONNX model saved to {onnx_path} with updated IR version {ONNX_IR_VERSION}")
    return onnx_path