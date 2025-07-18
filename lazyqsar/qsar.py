import os
import json
import numpy as np

from .descriptors import ChemeleonDescriptor
from .models import LazyRandomForestBinaryClassifier, LazyTuneTablesBinaryClassifier, LazyLogisticRegressionBinaryClassifier


descriptors_dict = {
    "chemeleon": ChemeleonDescriptor
}


models_dict = {
    "tune_tables": LazyTuneTablesBinaryClassifier,
    "random_forest": LazyRandomForestBinaryClassifier,
    "logistic_regression": LazyLogisticRegressionBinaryClassifier,
}

models_dict = dict((k, v) for k, v in models_dict.items() if v is not None)


class LazyBinaryQSAR(object):

    def __init__(self, descriptor_type="chemeleon", model_type="random_forest", **kwargs):
        self.descriptor_type = descriptor_type
        self.model_type = model_type

        if descriptor_type not in descriptors_dict:
            raise ValueError(f"Unsupported descriptor type: {descriptor_type}")
        self.descriptor = descriptors_dict[descriptor_type]()
        
        if model_type not in models_dict:
            raise ValueError(f"Unsupported model type: {model_type}")
        else:
            self.model = models_dict[model_type](**kwargs)

    def fit(self, X, y):
        y = np.array(y, dtype=int)
        print(f"Fitting inputs to feature descriptors using {self.descriptor_type}")
        self.descriptor.fit(X)
        print(f"Transforming inputs to feature descriptors using {self.descriptor_type}")
        descriptors = np.array(self.descriptor.transform(X), dtype=np.float32)
        print(f"Performing predictions on input feature of shape: {descriptors.shape}")
        self.model.fit(X=descriptors, y=y)

    def predict_proba(self, X):
        # Returns the probability of the positive class
        print(f"Transforming inputs to feature descriptors using {self.descriptor_type}")
        descriptors = np.array(self.descriptor.transform(X))
        print(f"Performing predictions on input feature of shape: {descriptors.shape}")
        return self.model.predict(descriptors)
    
    def save_model(self, model_dir: str):
        print(f"LazyQSAR Saving model to {model_dir}")
        config = {
            "descriptor_type": self.descriptor_type,
            "model_type": self.model_type,
        }
        self.model.save_model(model_dir)
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        metadata["descriptor_type"] = self.descriptor_type
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
        descriptor_type = config["descriptor_type"]
        model_type = config["model_type"]
        obj.descriptor_type = descriptor_type
        obj.model_type = model_type
        obj.descriptor = descriptors_dict[descriptor_type].load(model_dir)
        obj.model = models_dict[model_type].load_model(model_dir)
        print("Loading done!")
        return obj
