import json
import joblib
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

from ...utils.logging import logger

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

N_TRIALS = 10 # TODO increase for better tuning


def find_params(X, y):
    """
    Tune C for LogisticRegression with Optuna using out-of-fold ROC-AUC.
    Returns {"C": best_C}.
    """

    n_splits = 5
    random_state = 42
    max_iter = 1000
    n_trials = N_TRIALS

    logger.info("Finding best C for logistic regression head with Optuna...")
    X = np.asarray(X)
    y = np.asarray(y)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)

        oof = np.full(len(y), np.nan, dtype=np.float32)
        for tr, va in cv.split(X, y):
            clf = LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=random_state,
            )
            clf.fit(X[tr], y[tr])
            oof[va] = clf.predict_proba(X[va])[:, 1].astype(np.float32)

        if np.isnan(oof).any():
            return 0.5

        auc = roc_auc_score(y, oof)
        trial.report(auc, step=0)
        return auc

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    study.enqueue_trial({"C": 1.0})
    study.optimize(objective, n_trials=n_trials)

    best_C = float(study.best_params["C"])
    logger.info(f"Best C: {best_C}")
    return {"C": best_C}



class Head(BaseEstimator, ClassifierMixin):

    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        logger.info("Fitting logistic regression head...")
        self.model = LogisticRegression(C=self.C, class_weight="balanced")
        self.model.fit(X, y)
        self.calibrate(X, y)
        self.input_dim = X.shape[1]
        return self

    def calibrate(self, X, y):
        logger.info("Evaluating logistic regression head...")
        splitter = StratifiedKFold(n_splits=5, shuffle=True)
        y_hat = []
        y_true = []
        for train_idx, test_idx in splitter.split(X, y):
            self.model.fit(X[train_idx], y[train_idx])
            y_hat_fold = self.model.predict_proba(X[test_idx])[:, 1]
            y_hat += list(y_hat_fold)
            y_true += list(y[test_idx])
        self.calibrator = LogisticRegression(class_weight="balanced")
        self.calibrator.fit(np.array(y_hat).reshape(-1, 1), y_true)
        self.score = roc_auc_score(y_true, y_hat)
        logger.info(f"ROC-AUC: {self.score}")
        return self.score

    def predict_proba(self, X):
        y_hat = self.model.predict_proba(X)[:, 1]
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
    