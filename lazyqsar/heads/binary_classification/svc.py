import json
import joblib
import os
import numpy as np
import optuna
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

from ...utils.logging import logger

N_TRIALS = 10 # TODO increase for better tuning


def find_params(X, y, timeout=None, random_state=42):
    """
    Tune C for LinearSVC with Optuna using out-of-fold ROC-AUC on decision_function.

    Returns
    -------
    dict: {"C": best_C}
    """
    logger.info("Finding best C for SVC head...")
    X = np.asarray(X)
    y = np.asarray(y)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    n_trials = N_TRIALS

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)

        oof = np.full(len(y), np.nan, dtype=np.float32)
        for tr, va in kf.split(X, y):
            clf = LinearSVC(
                C=C,
                random_state=random_state,
            )
            clf.fit(X[tr], y[tr])
            oof[va] = clf.decision_function(X[va]).astype(np.float32)

        if np.isnan(oof).any():
            return 0.5

        return roc_auc_score(y, oof)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.enqueue_trial({"C": 1.0})
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return {"C": float(study.best_params["C"])}


class Head(BaseEstimator, ClassifierMixin):

    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        logger.info("Fitting SVC head...")
        self.model = LinearSVC(C=self.C, class_weight="balanced")
        self.model.fit(X, y)
        self.calibrate(X, y)
        self.input_dim = X.shape[1]
        return self

    def calibrate(self, X, y):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_hat = []
        y_true = []
        for train_idx, test_idx in splitter.split(X, y):
            self.model.fit(X[train_idx], y[train_idx])
            y_hat_fold = self.model.decision_function(X[test_idx]).astype(np.float32)
            y_hat += list(y_hat_fold)
            y_true += list(y[test_idx])
        self.calibrator = LogisticRegression().fit(np.array(y_hat).reshape(-1, 1), np.array(y_true))
        self.score = roc_auc_score(y_true, y_hat)

    def predict_proba(self, X):
        y_hat = self.model.decision_function(X).astype(np.float32)
        y_hat = self.calibrator.predict_proba(y_hat.reshape(-1, 1))[:, 1]
        return np.vstack([1 - y_hat, y_hat]).T

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, name: str, model_dir: str):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        metadata = {
            "C": self.C,
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
        head = cls(C=metadata["C"])
        head.model = model
        head.calibrator = calibrator
        head.score = metadata["score"]
        head.input_dim = metadata["input_dim"]
        return head