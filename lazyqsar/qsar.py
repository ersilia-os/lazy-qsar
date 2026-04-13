import hashlib
import os
import shutil
import numpy as np

from .utils.logging import logger


DESCRIPTOR_TYPES = {
    "chemeleon": ("lazyqsar.descriptors.chemeleon", "ChemeleonDescriptor"),
    "morgan": ("lazyqsar.descriptors.morgan", "MorganFingerprint"),
    "rdkit": ("lazyqsar.descriptors.rdkit_descriptors", "RDKitDescriptor"),
    "cddd": ("lazyqsar.descriptors.cddd", "ContinuousDataDrivenDescriptor"),
}

DESCRIPTORS_MODE = {
    "default": ["chemeleon", "rdkit", "cddd"],
    "fast": ["rdkit", "morgan"],
    "slow": ["chemeleon", "morgan", "rdkit", "cddd"],
}

DESCRIPTORS_MODE = {k: sorted(v) for k, v in DESCRIPTORS_MODE.items()}


def get_descriptor_type(descriptor_name):
    module_name, class_name = DESCRIPTOR_TYPES[descriptor_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)



class ArtifactWrapper(object):
    def __init__(self, descriptors, artifacts, weights):
        self.descriptors = descriptors
        self.artifacts = artifacts
        self.weights = weights

    def predict_proba(self, smiles_list):
        y_hats = []
        for descriptor, artifact in zip(self.descriptors, self.artifacts):
            X = descriptor.transform(smiles_list)
            y_hats.append(np.array(artifact.predict_proba(X))[:, 1])
        y_hat_1 = np.average(np.array(y_hats), axis=0, weights=self.weights)
        return np.vstack((1 - y_hat_1, y_hat_1)).T

    def predict(self, smiles_list, cutoff=0.5):
        y_hat = self.predict_proba(smiles_list)[:, 1]
        return (y_hat >= cutoff).astype(int)


class LazyClassifierQSAR(object):
    def __init__(
        self,
        mode: str = "default",
    ):
        assert mode in ["default", "fast", "slow"], (
            f"Mode {mode} not recognized. Choose from 'default', 'fast', or 'slow'."
        )

        descriptor_types = DESCRIPTORS_MODE[mode]

        self.mode = mode
        self.descriptor_types = descriptor_types
        self.descriptors = [
            get_descriptor_type(descriptor_type)() for descriptor_type in descriptor_types
        ]
        self.is_saved = False
        self.weights = None
        self._feature_cache = {}  # {(descriptor_idx, smiles_hash): X}
        self._n_train_samples = None

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
        n = len(self.models)
        self.weights = np.ones(n) / n

    def fit(self, smiles_list, y):
        from .agnostic import LazyClassifier

        y = np.array(y, dtype=int)
        Xs = [self._transform_cached(i, smiles_list) for i in range(len(self.descriptors))]
        self.models = []
        for i in range(len(self.descriptors)):
            logger.info(f"Fitting with descriptor: {self.descriptor_types[i]}")
            model = LazyClassifier()
            model.fit(X=Xs[i], y=y)
            self.models += [model]
        self._assign_weights()

    def predict_proba(self, smiles_list):
        y_hats = []
        for i in range(len(self.descriptors)):
            X = self._transform_cached(i, smiles_list)
            y_hats.append(self.models[i].predict_proba(X=X)[:, 1])
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
        from .agnostic import LazyClassifier

        descriptors = []
        models = []
        for descriptor_type in descriptor_types:
            model_subdir = os.path.join(model_dir, descriptor_type)
            if not os.path.exists(model_subdir):
                raise FileNotFoundError(
                    f"Descriptor directory {model_subdir} does not exist."
                )
            descriptors += [get_descriptor_type(descriptor_type).load(model_subdir)]
            models += [LazyClassifier.load(model_subdir)]

        obj = cls(mode=mode)
        obj.descriptors = descriptors
        obj.models = models
        obj._n_train_samples = None
        obj._assign_weights()
        obj.is_saved = True
        return obj

    def save_onnx(self, model_dir: str, clean: bool = True):
        # ONNX is already written by save_raw() via LazyClassifier.save().
        pass

    @classmethod
    def load_onnx(cls, model_dir: str):
        descriptor_types = []
        for fn in os.listdir(model_dir):
            if fn in DESCRIPTOR_TYPES.keys():
                descriptor_types += [fn]
        descriptor_types = sorted(descriptor_types)
        from .agnostic import LazyClassifier

        descriptors = []
        artifacts = []
        for descriptor_type in descriptor_types:
            model_subdir = os.path.join(model_dir, descriptor_type)
            if not os.path.exists(model_subdir):
                raise FileNotFoundError(
                    f"Descriptor directory {model_subdir} does not exist."
                )
            descriptors += [get_descriptor_type(descriptor_type).load(model_subdir)]
            artifacts += [LazyClassifier.load(model_subdir)]
        n = len(artifacts)
        weights = np.ones(n) / n
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


class LazyRegressorQSAR:
    """Placeholder — not yet implemented."""

    def __init__(self, mode: str = "default"):
        raise NotImplementedError("LazyRegressorQSAR is not yet implemented.")


class LazyQSAR:
    """
    Dispatcher that returns the appropriate QSAR class based on task.

    LazyQSAR(task='classification', **kwargs)  →  LazyClassifierQSAR(**kwargs)
    LazyQSAR(task='regression', **kwargs)      →  LazyRegressorQSAR(**kwargs)
    """

    def __new__(cls, task: str = "classification", **kwargs):
        if task == "classification":
            return LazyClassifierQSAR(**kwargs)
        elif task == "regression":
            return LazyRegressorQSAR(**kwargs)
        else:
            raise ValueError(
                f"Unknown task {task!r}. Choose 'classification' or 'regression'."
            )
