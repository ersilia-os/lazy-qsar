import os
import json
import numpy as np

from .models import (
    LazyRandomForestBinaryClassifier,
    LazyTuneTablesBinaryClassifier,
    LazyLogisticRegressionBinaryClassifier,
)


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
    def __init__(self, model_type="logistic_regression", **kwargs):
        self.model_type = model_type

        if model_type not in binary_models_dict:
            print(binary_models_dict)
            raise ValueError(f"Unsupported model type: {model_type}")
        else:
            self.model = binary_models_dict[model_type](**kwargs)

    def fit(self, X=None, y=None, h5_file=None, h5_idxs=None):
        y = np.array(y, dtype=int)
        self.model.fit(X=X, y=y, h5_file=h5_file, h5_idxs=h5_idxs)

    def predict_proba(self, X=None, h5_file=None, h5_idxs=None):
        return self.model.predict(X=X, h5_file=h5_file, h5_idxs=h5_idxs)

    def save_model(self, model_dir: str):
        print(f"LazyQSAR Saving model to {model_dir}")
        config = {
            "model_type": self.model_type,
        }
        self.model.save_model(model_dir)
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        metadata["model_type"] = self.model_type
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)
        print("Saving done!")

    @classmethod
    def load_model(cls, model_dir: str):
        print(f"LazyQSAR Loading model from {model_dir}")
        obj = cls()
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        model_type = config["model_type"]
        obj.model_type = model_type
        obj.model = binary_models_dict[model_type].load_model(model_dir)
        print("Loading done!")
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
        print(f"LazyQSAR Saving model to {model_dir}")
        config = {
            "model_type": self.model_type,
        }
        self.model.save_model(model_dir)
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        metadata["model_type"] = self.model_type
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)
        print("Saving done!")

    @classmethod
    def load_model(cls, model_dir: str):
        print(f"LazyQSAR Loading model from {model_dir}")
        obj = cls()
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        model_type = config["model_type"]
        obj.model_type = model_type
        obj.model = regression_models_dict[model_type].load_model(model_dir)
        print("Loading done!")
        return obj
