import os
import json
import shutil
import numpy as np

from .descriptors.chemeleon import ChemeleonDescriptor
from .descriptors.morgan import MorganFingerprint
from .descriptors.rdkit_descriptors import RDKitDescriptor
from .descriptors.cddd import ContinuousDataDrivenDescriptor

from .agnostic import LazyEclecticBinaryClassifier
from .agnostic import LazyBinaryClassifierArtifact
from .agnostic import convert_to_onnx
from .agnostic import NUM_TRIALS_MODES

from .utils.logging import logger


DESCRIPTOR_TYPES = {
    "chemeleon": ChemeleonDescriptor,
    "morgan": MorganFingerprint,
    "rdkit": RDKitDescriptor,
    "cddd": ContinuousDataDrivenDescriptor,
}

DESCRIPTORS_MODE = {
    "default": ["chemeleon", "rdkit", "cddd"],
    "fast": ["rdkit", "morgan"],
    "slow": ["chemeleon", "morgan", "rdkit", "cddd"],
}

DESCRIPTORS_MODE = {k: sorted(v) for k, v in DESCRIPTORS_MODE.items()}


class ArtifactWrapper(object):
    def __init__(self, descriptors, artifacts, weights):
        self.descriptors = descriptors
        self.artifacts = artifacts
        self.weights = weights

    def predict_proba(self, smiles_list):
        R = []
        for descriptor, artifact in zip(self.descriptors, self.artifacts):
            X = descriptor.transform(smiles_list)
            y_hat_1 = np.array(artifact.predict_proba(X))[:, 1]
            R += [y_hat_1]
        y_hat_1 = np.average(np.array(R), axis=0, weights=self.weights)
        y_hat_0 = 1 - y_hat_1
        return np.vstack((y_hat_0, y_hat_1)).T

    def predict(self, smiles_list, cutoff=0.5):
        y_hat = self.predict_proba(smiles_list)[:, 1]
        return (y_hat >= cutoff).astype(int)


class LazyBinaryQSAR(object):
    def __init__(self, mode: str = "default"):
        assert mode in ["default", "fast", "slow"], (
            f"Mode {mode} not recognized. Choose from 'default', 'fast', or 'slow'."
        )

        descriptor_types = DESCRIPTORS_MODE[mode]

        self.descriptor_types = descriptor_types
        self.descriptors = [
            DESCRIPTOR_TYPES[descriptor_type]() for descriptor_type in descriptor_types
        ]
        self.num_trials = NUM_TRIALS_MODES[mode]
        self.is_saved = False
        self.weights = None

    def _assign_weights(self):
        scores = []
        for m in self.models:
            scores += [m.score]
        weights = np.clip(np.array(scores) - 0.5, a_min=0, a_max=1) + 1e-4
        weights = weights / np.sum(weights)
        self.weights = weights

    def fit(self, smiles_list, y):
        y = np.array(y, dtype=int)
        self.models = []
        for i, descriptor in enumerate(self.descriptors):
            logger.info(f"Fitting with descriptor: {self.descriptor_types[i]}")
            X = descriptor.transform(smiles_list)
            mask = ~np.isnan(X).any(axis=1)
            model = LazyEclecticBinaryClassifier(num_trials=self.num_trials)
            model.fit(X=X[mask], y=y[mask])
            self.models += [model]
        self._assign_weights()

    def predict_proba(self, smiles_list):
        R = []
        for i, descriptor in enumerate(self.descriptors):
            X = descriptor.transform(smiles_list)
            y_hat_1 = np.full(shape=(X.shape[0],), fill_value=np.nan)
            mask = ~np.isnan(X).any(axis=1)
            y_hat_1[mask] = np.array(self.models[i].predict(X=X[mask]))
            R += [y_hat_1]
        R = np.array(R)
        y_hat_1 = []
        for i in range(R.shape[0]):
            row = R[:, i]
            mask = ~np.isnan(row)
            weights = self.weights[mask]/np.sum(self.weights[mask])
            y_hat_1 += [np.average(row[mask], weights=weights)]
        y_hat_1 = np.array(y_hat_1)
        y_hat_0 = 1 - y_hat_1
        return np.vstack((y_hat_0, y_hat_1)).T

    def predict(self, smiles_list, threshold=0.5):
        y_hat = self.predict_proba(smiles_list)[:, 1]
        y_bin = []
        for y in y_hat:
            if y >= threshold:
                y_bin.append(1)
            else:
                y_bin.append(0)
        return np.array(y_bin, dtype=int)

    def save_raw(self, model_dir: str):
        for i, descriptor_name in enumerate(self.descriptor_types):
            model_subdir = os.path.join(model_dir, descriptor_name)
            if not os.path.exists(model_subdir):
                os.makedirs(model_subdir)
            logger.debug(f"Saving model to {model_subdir}")
            self.models[i].save(model_subdir)
            logger.debug(f"Saving descriptor to {model_subdir}")
            self.descriptors[i].save(model_subdir)
        self.is_saved = True

    @classmethod
    def load_raw(cls, model_dir: str):
        descriptor_types = []
        for fn in os.listdir(model_dir):
            if fn in DESCRIPTOR_TYPES.keys():
                descriptor_types += [fn]
        descriptor_types = sorted(descriptor_types)
        mode = None
        for k, v in DESCRIPTORS_MODE.items():
            if set(v) == set(descriptor_types):
                mode = k
                break
        if mode is None:
            raise Exception(
                "Could not infer mode from descriptor types found in the model directory."
            )
        descriptors = []
        models = []
        for descriptor_type in descriptor_types:
            model_subdir = os.path.join(model_dir, descriptor_type)
            if not os.path.exists(model_subdir):
                raise FileNotFoundError(
                    f"Descriptor directory {model_subdir} does not exist."
                )
            with open(os.path.join(model_subdir, "metadata.json"), "r") as f:
                metadata = json.load(f)
            num_trials = metadata["num_trials"]
            mode_ = None
            for k, v in NUM_TRIALS_MODES.items():
                if v == num_trials:
                    mode_ = k
            if mode_ is None or mode_ != mode:
                raise Exception("Inconsistent mode found in model directories.")
            descriptors += [DESCRIPTOR_TYPES[descriptor_type].load(model_subdir)]
            models += [LazyEclecticBinaryClassifier.load(model_subdir)]

        obj = cls(mode=mode)
        obj.descriptors = descriptors
        obj.models = models
        obj._assign_weights()
        obj.is_saved = True
        return obj

    def save_onnx(self, model_dir: str, clean: bool = True):
        if not self.is_saved:
            self.save(model_dir)
        descriptor_types = []
        for fn in os.listdir(model_dir):
            if fn in DESCRIPTOR_TYPES.keys():
                descriptor_types += [fn]
        descriptor_types = sorted(descriptor_types)
        for descriptor_type in descriptor_types:
            model_subdir = os.path.join(model_dir, descriptor_type)
            convert_to_onnx(model_subdir, clean=clean)

    @classmethod
    def load_onnx(cls, model_dir: str):
        descriptor_types = []
        for fn in os.listdir(model_dir):
            if fn in DESCRIPTOR_TYPES.keys():
                descriptor_types += [fn]
        descriptor_types = sorted(descriptor_types)
        descriptors = []
        artifacts = []
        scores = []
        for descriptor_type in descriptor_types:
            model_subdir = os.path.join(model_dir, descriptor_type)
            if not os.path.exists(model_subdir):
                raise FileNotFoundError(
                    f"Descriptor directory {model_subdir} does not exist."
                )
            descriptors += [DESCRIPTOR_TYPES[descriptor_type].load(model_subdir)]
            artifacts += [LazyBinaryClassifierArtifact.load(model_dir=model_subdir)]
            metadata = {}
            with open(os.path.join(model_subdir, "metadata.json"), "r") as f:
                metadata = json.load(f)
                scores += [metadata["score"]]
        weights = np.clip(np.array(scores) - 0.5, a_min=0, a_max=1) + 1e-4
        weights = weights / np.sum(weights)
        return ArtifactWrapper(
            descriptors=descriptors, artifacts=artifacts, weights=weights
        )

    def save(self, model_dir: str, onnx: bool = True):
        if model_dir.endswith(".zip"):
            zip = True
            model_dir = model_dir[:-4]
        else:
            zip = False
        self.save_raw(model_dir)
        if onnx:
            self.save_onnx(model_dir)
        if zip:
            shutil.make_archive(model_dir, "zip", model_dir)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            return model_dir + ".zip"
        return model_dir

    @classmethod
    def load(cls, model_dir: str):
        if model_dir.endswith(".zip"):
            zip = True
        else:
            zip = False
        if zip:
            base_dir = model_dir[:-4]
            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)
            shutil.unpack_archive(model_dir, base_dir)
            model_dir = base_dir
        descriptor_types = []
        for fn in os.listdir(model_dir):
            if fn in DESCRIPTOR_TYPES.keys():
                descriptor_types += [fn]
        descriptor_types = sorted(descriptor_types)
        for descriptor_type in descriptor_types:
            model_subdir = os.path.join(model_dir, descriptor_type)
            for fn in os.listdir(model_subdir):
                if fn.endswith(".onnx"):
                    return cls.load_onnx(model_dir=model_dir)
        obj = cls.load_raw(model_dir=model_dir)
        if zip:
            shutil.rmtree(base_dir)
        return obj
