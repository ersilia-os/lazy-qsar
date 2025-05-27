import os
import json
import joblib

from .descriptors import MorganDescriptor, RDKitDescriptor, ClassicDescriptor, ErsiliaEmbeddingEmbedding, MaccsDescriptor
from .models import LazyXGBoostBinaryClassifier


descriptors_dict = {
    "morgan": MorganDescriptor,
    "rdkit": RDKitDescriptor,
    "classic": ClassicDescriptor,
    "ersilia_embedding": ErsiliaEmbeddingEmbedding,
    "maccs": MaccsDescriptor,
}


models_dict = {
    "xgboost": LazyXGBoostBinaryClassifier,
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
        self.descriptor.fit(X)
        descriptors = self.descriptor.transform(X)
        self.model.fit(descriptors, y)

    def predict(self, X):
        descriptors = self.descriptor.transform(X)
        return self.model.predict(descriptors)
    
    def save_model(self, model_dir: str):
        config = {
            "descriptor_type": self.descriptor_type,
            "model_type": self.model_type,
        }
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)
        self.model.save_model(model_dir)

    @classmethod
    def load_model(cls, model_dir: str):

        obj = cls()

        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        descriptor_type = config["descriptor_type"]
        model_type = config["model_type"]
        obj.descriptor_type = descriptor_type
        obj.model_type = model_type
        obj.descriptor = joblib.load(os.path.join(model_dir, "descriptor.joblib"))
        obj.model = models_dict[model_type].load_model(model_dir)
       
        return obj