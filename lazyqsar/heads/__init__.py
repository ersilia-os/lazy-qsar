"""The heads an assembler fits: one model family for one task, behind one interface.

``Head(task, name, **kwargs)`` builds the base estimator registered for ``(task, name)``
and exposes what the assemblers call. Classification heads are ``lr``, ``xgb``, ``rf``
and ``svc``; regression heads are ``lr``, ``xgb``, ``rf`` and ``svr``.
"""

from __future__ import annotations

import importlib

from lazyqsar.utils.logging import logger

# (task, head name) -> (module, class)
HEAD_CLASSES = {
    ("classification", "lr"): ("lazyqsar.base.linear", "BaseLinearClassifier"),
    ("classification", "xgb"): ("lazyqsar.base.xgboost", "BaseXGBClassifier"),
    ("classification", "rf"): ("lazyqsar.base.randomforest", "BaseRFClassifier"),
    ("classification", "svc"): ("lazyqsar.base.svc", "BaseSVCClassifier"),
    ("regression", "lr"): ("lazyqsar.base.linear", "BaseLinearRegressor"),
    ("regression", "xgb"): ("lazyqsar.base.xgboost", "BaseXGBRegressor"),
    ("regression", "rf"): ("lazyqsar.base.randomforest", "BaseRFRegressor"),
    ("regression", "svr"): ("lazyqsar.base.svc", "BaseSVR"),
}

HEAD_LABELS = {"lr": "LR", "xgb": "XGB", "rf": "RF", "svc": "SVC", "svr": "SVR"}


def head_kwargs(task: str, name: str, calibrated: bool, max_rounds) -> dict:
    """Constructor arguments for a head: the XGBoost round cap, calibration (classifiers)."""
    kwargs = {}
    if task == "classification":
        kwargs["calibrated"] = calibrated
    if name == "xgb":
        kwargs["max_rounds"] = max_rounds
    return kwargs


class Head:
    """A fitted-or-not base estimator for one ``(task, name)``; ``model`` is the estimator."""

    def __init__(self, task: str, name: str, **kwargs):
        try:
            module, cls_name = HEAD_CLASSES[(task, name)]
        except KeyError:
            raise ValueError(f"Unknown {task} head {name!r}.") from None
        self.task = task
        self.name = name
        self.label = HEAD_LABELS.get(name, name)
        self.model = getattr(importlib.import_module(module), cls_name)(**kwargs)

    def fit(self, X, y):
        logger.debug(f"Fitting {self.label} head | X={X.shape}")
        self.model.fit(X, y)
        preset = (
            getattr(self.model, "preset_name_", None)
            or getattr(self.model, "selected_preset_", None)
            or getattr(self.model, "regime_", None)
        )
        logger.debug(f"{self.label} head fitted | preset={preset}")

    def predict(self, X, *args, **kwargs):
        return self.model.predict(X, *args, **kwargs)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_score(self, X):
        return self.model.predict_score(X)

    def _oof_percentile(self, X):
        return self.model.predict_rank(X)

    def save(self, directory):
        self.model.save(directory)
        logger.debug(f"{self.label} head saved to {directory}")
