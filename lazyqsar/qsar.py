import os
import json
import joblib
import numpy as np

from .descriptors import MorganDescriptor, MordredDescriptor, RdkitDescriptor, ClassicDescriptor, MaccsDescriptor
from .models import LazyXGBoostBinaryClassifier, TuneTablesClassifierLight


descriptors_dict = {
    "morgan": MorganDescriptor,
    "mordred": MordredDescriptor,
    "rdkit": RdkitDescriptor,
    "classic": ClassicDescriptor,
    "maccs": MaccsDescriptor,
}


models_dict = {
    "xgboost": LazyXGBoostBinaryClassifier,
    "tunetables": TuneTablesClassifierLight,
}


class LazyBinaryQSAR(object):

    def __init__(self, descriptor_type="morgan", model_type="xgboost", **kwargs):
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
        self.descriptor.fit(X)
        descriptors = np.array(self.descriptor.transform(X))
        self.model.fit(descriptors, y)

    def predict(self, X):
        descriptors = np.array(self.descriptor.transform(X))
        return self.model.predict(descriptors)
    
    def predict_proba(self, X):
        descriptors = np.array(self.descriptor.transform(X))
        return self.model.predict_proba(descriptors)
    
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

