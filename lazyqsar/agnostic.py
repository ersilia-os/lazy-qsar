import os
import json
import numpy as np

from .models import (
    LazyRandomForestBinaryClassifier,
    LazyTuneTablesBinaryClassifier,
    LazyLogisticRegressionBinaryClassifier,
)

from .config.presets import preset_params
from .utils.logging import logger

binary_models_dict = {
    "tune_tables": LazyTuneTablesBinaryClassifier,
    "random_forest": LazyRandomForestBinaryClassifier,
    "logistic_regression": LazyLogisticRegressionBinaryClassifier,
}

binary_models_dict = dict(
    (k, v) for k, v in binary_models_dict.items() if v is not None
)

regression_models_dict = {"linear_model": None}


class LazyBinaryClassifier(object):
    def __init__(self, model_type: str = "random_forest", mode: str = "default", **kwargs):
        """
        Initialize a LazyBinaryClassifier

        This class serves as a wrapper for various binary classification models,
        allowing for easy switching between different algorithms and configurations.
        Args:
            model_type (str): The type of model to use. Options are 'random_forest' (default), 'logistic_regression' or 'tune_tables'.
            mode (str): The preset mode to use for the model parameters. Options are 'quick' or 'default'.
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Usage:
        >>> from lazyqsar.models import LazyBinaryClassifier
        >>> classifier = LazyBinaryClassifier(model_type="random_forest", mode="default")
        >>> classifier.fit(X=X, y=y)
        >>> predictions = classifier.predict_proba(X=X)
        """

        self.model_type = model_type
        self.mode = mode

        if mode not in preset_params:
            raise ValueError(f"Unsupported mode: {mode}. Please choose from {list(preset_params.keys())}.")

        presets = preset_params[mode]
        combined_kwargs = {**presets, **kwargs}

        if model_type not in binary_models_dict:
            raise ValueError(f"Unsupported model type: {model_type}")
        else:
            self.model = binary_models_dict[model_type](**combined_kwargs)

    def fit(self, X=None, y=None, h5_file=None, h5_idxs=None):
        y = np.array(y, dtype=int)
        self.model.fit(X=X, y=y, h5_file=h5_file, h5_idxs=h5_idxs)

    def predict_proba(self, X=None, h5_file=None, h5_idxs=None):
        y_hat_1 = np.array(self.model.predict(X=X, h5_file=h5_file, h5_idxs=h5_idxs))
        y_hat_0 = 1 - y_hat_1
        return np.array([y_hat_0, y_hat_1]).T
    
    def predict(self, X=None, h5_file=None, h5_idxs=None, threshold=0.5):
        y_hat = self.predict_proba(X=X, h5_file=h5_file, h5_idxs=h5_idxs)[:, 1]
        y_bin = []
        for y in y_hat:
            if y >= threshold:
                y_bin.append(1)
            else:
                y_bin.append(0)
        return np.array(y_bin, dtype=int)

    def save_model(self, model_dir: str):
        logger.debug(f"LazyQSAR Saving model to {model_dir}")
        config = {
            "model_type": self.model_type,
        }
        self.model.save_model(model_dir)
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        metadata["model_type"] = self.model_type
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)
        logger.info(f"The model is successfully saved at {model_dir}")

    @classmethod
    def load_model(cls, model_dir: str):
        logger.debug(f"LazyQSAR Loading model from {model_dir}")
        obj = cls()
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        model_type = config["model_type"]
        obj.model_type = model_type
        obj.model = binary_models_dict[model_type].load_model(model_dir)
        logger.info(f"Model successfully loaded from {model_dir}")
        return obj


class LazyRegressor(object):
    def __init__(self, model_type="logistic_regression", **kwargs):
        self.model_type = model_type

        if model_type not in regression_models_dict:
            raise ValueError(f"Unsupported model type: {model_type}")
        else:
            self.model = regression_models_dict[model_type](**kwargs)

    def fit(self, X, y):
        y = np.array(y, dtype=float)
        if isinstance(X[0], str):
            raise ValueError(
                "The input X can not be a string! Transfer it to the descriptors!"
            )
        self.model.fit(X=X, y=y)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, model_dir: str):
        logger.debug(f"LazyQSAR Saving model to {model_dir}")
        config = {
            "model_type": self.model_type,
        }
        self.model.save_model(model_dir)
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        metadata["model_type"] = self.model_type
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)
        logger.info(f"The model is successfully saved at {model_dir}")

    @classmethod
    def load_model(cls, model_dir: str):
        logger.debug(f"LazyQSAR Loading model from {model_dir}")
        obj = cls()
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        model_type = config["model_type"]
        obj.model_type = model_type
        obj.model = regression_models_dict[model_type].load_model(model_dir)
        logger.info(f"Model successfully loaded from {model_dir}")
        return obj
