import hashlib
import os
import json
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .descriptors.chemeleon import ChemeleonDescriptor
from .descriptors.morgan import MorganFingerprint
from .descriptors.rdkit_descriptors import RDKitDescriptor
from .descriptors.cddd import ContinuousDataDrivenDescriptor

from .agnostic import LazyEclecticBinaryClassifier
from .agnostic import LazyBinaryClassifierArtifact
from .agnostic import convert_to_onnx

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
        n = len(self.descriptors)
        y_hats = [None] * n

        def _predict_one(i):
            X = self.descriptors[i].transform(smiles_list)
            return np.array(self.artifacts[i].predict_proba(X))[:, 1]

        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {ex.submit(_predict_one, i): i for i in range(n)}
            for future in as_completed(futures):
                y_hats[futures[future]] = future.result()

        y_hat_1 = np.average(np.array(y_hats), axis=0, weights=self.weights)
        return np.vstack((1 - y_hat_1, y_hat_1)).T

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
        self.is_saved = False
        self.weights = None
        self._feature_cache = {}  # {(descriptor_idx, smiles_hash): X}

    def _smiles_hash(self, smiles_list):
        return hashlib.md5("\x00".join(smiles_list).encode()).hexdigest()

    def _transform_cached(self, i, smiles_list):
        key = (i, self._smiles_hash(smiles_list))
        if key not in self._feature_cache:
            self._feature_cache[key] = self.descriptors[i].transform(smiles_list)
        else:
            logger.debug(f"Using cached features for descriptor: {self.descriptor_types[i]}")
        return self._feature_cache[key]

    def _assign_weights(self):
        scores = []
        for m in self.models:
            scores += [m.score]
        weights = np.clip(np.array(scores) - 0.5, a_min=0, a_max=1) + 1e-4
        weights = weights / np.sum(weights)
        self.weights = weights

    def fit(self, smiles_list, y):
        y = np.array(y, dtype=int)
        n = len(self.descriptors)
        Xs = [None] * n
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {ex.submit(self._transform_cached, i, smiles_list): i for i in range(n)}
            for future in as_completed(futures):
                Xs[futures[future]] = future.result()
        self.models = []
        for i in range(n):
            logger.info(f"Fitting with descriptor: {self.descriptor_types[i]}")
            model = LazyEclecticBinaryClassifier()
            model.fit(X=Xs[i], y=y)
            self.models += [model]
        self._assign_weights()

    def predict_proba(self, smiles_list):
        n = len(self.descriptors)
        y_hats = [None] * n

        def _predict_one(i):
            X = self._transform_cached(i, smiles_list)
            return np.array(self.models[i].predict(X=X))

        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {ex.submit(_predict_one, i): i for i in range(n)}
            for future in as_completed(futures):
                y_hats[futures[future]] = future.result()

        y_hat_1 = np.average(np.array(y_hats), axis=0, weights=self.weights)
        return np.vstack((1 - y_hat_1, y_hat_1)).T

    def predict(self, smiles_list, threshold=0.5):
        return (self.predict_proba(smiles_list)[:, 1] >= threshold).astype(int)

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
