import os
import json
import numpy as np

from .models import LazyXGBoostBinaryClassifier, TuneTablesBinaryClassifier, TuneTablesZeroShotBinaryClassifier, LazyZSRandomForestBinaryClassifier, LazyRandomForestBinaryClassifier


models_dict = {
    "xgboost": LazyXGBoostBinaryClassifier,
    "tunetables": TuneTablesBinaryClassifier,
    "zstunetables": TuneTablesZeroShotBinaryClassifier,
    "zsrandomforest": LazyZSRandomForestBinaryClassifier,
    "randomforest":LazyRandomForestBinaryClassifier
}


class LazyBinaryQSAR(object):

    def __init__(self, model_type="zsrandomforest", **kwargs):
        self.model_type = model_type

        if model_type not in models_dict:
            raise ValueError(f"Unsupported model type: {model_type}")
        else:
            self.model = models_dict[model_type](**kwargs)

    def fit(self, X, y):
        y = np.array(y, dtype=int)
        if isinstance(X[0], str):
            raise ValueError("The input X can not be a string! Transfor it to the descriptors!")
        self.model.fit(X=X, y=y)

    def predict_proba(self, X):
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
        self.descriptor.save(model_dir)
        print("Saving done!")

    @classmethod
    def load_model(cls, model_dir: str):
        print(f"LazyQSAR Loading model from {model_dir}")
        obj = cls()
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        model_type = config["model_type"]
        obj.model_type = model_type
        obj.model = models_dict[model_type].load_model(model_dir)
        print("Loading done!")
        return obj
