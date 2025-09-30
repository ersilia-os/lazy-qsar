import os
import json
import numpy as np

from .descriptors import ChemeleonDescriptor, MorganFingerprint

from .agnostic import LazyEclecticBinaryClassifier
from .agnostic import LazyBinaryClassifierArtifact
from .agnostic import convert_to_onnx
from .agnostic import NUM_TRIALS_MODES


DESCRIPTOR_TYPES = {
    "chemeleon": ChemeleonDescriptor,
    "morgan": MorganFingerprint
}


class ArtifactWrapper(object):

    def __init__(self, descriptor, artifact):
        self.descriptor = descriptor
        self.artifact = artifact

    def predict_proba(self, smiles_list):
        X = self.descriptor.transform(smiles_list)
        return self.artifact.predict_proba(X)
    
    def predict(self, smiles_list):
        X = self.descriptor.transform(smiles_list)
        return self.artifact.predict(X)


class LazyBinaryQSAR(object):

    def __init__(self, descriptor_type, mode):
        self.descriptor_type = descriptor_type
        self.descriptor = DESCRIPTOR_TYPES[descriptor_type]
        self.num_trials = NUM_TRIALS_MODES[mode]
        self.is_saved = False

    def fit(self, smiles_list, y):
        y = np.array(y, dtype=int)
        self.descriptor.fit(smiles_list)
        descriptors = self.descriptor.transform(smiles_list)
        self.model = LazyEclecticBinaryClassifier(num_trials=self.num_trials)
        self.model.fit(X=descriptors, y=y)

    def predict_proba(self, smiles_list):
        X = self.descriptor.transform(smiles_list)
        y_hat_1 = np.array(self.model.predict(X=X))
        y_hat_0 = 1 - y_hat_1
        return np.array([y_hat_0, y_hat_1]).T

    def predict(self, smiles_list, threshold=0.5):
        y_hat = self.predict_proba(smiles_list)[:,1]
        y_bin = []
        for y in y_hat:
            if y >= threshold:
                y_bin.append(1)
            else:
                y_bin.append(0)
        return np.array(y_bin, dtype=int)

    def save(self, model_dir: str):
        self.model.save(model_dir)
        metadata = {
            "descriptor_type": self.descriptor_type
        }
        with open(os.path.join(model_dir, "descriptor.json"), "w") as f:
            json.dump(metadata, f)
        self.is_saved = True

    @classmethod
    def load(cls, model_dir: str):
        with open(os.path.join(model_dir, "descriptor.json"), "r") as f:
            desc_metadata = json.load(f)
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        num_trials = metadata["num_trials"]
        mode = None
        for k,v in NUM_TRIALS_MODES.keys():
            if v == num_trials:
                mode = k
        obj = cls(descriptor_type=desc_metadata["descriptor_type"], mode=mode)
        obj.model = LazyEclecticBinaryClassifier.load(model_dir)
        obj.is_saved = True
        return obj

    def save_onnx(self, model_dir: str, clean: bool = True):
        if not self.is_saved:
            self.save(model_dir)
        convert_to_onnx(model_dir, clean=clean)
        # TODO need to skip "descriptor.json" in the cleaning

    def load_onnx(self, model_dir: str):
        art = LazyBinaryClassifierArtifact.load(model_dir=model_dir)
        # TODO need to figure out the fit of the descriptor