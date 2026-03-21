import os
import shutil
import numpy as np

from .assemblers.eclectic_binary_classifier import (
    LazyEclecticBinaryClassifier,
    convert_to_onnx,
)
from .artifacts.artifact_binary_classifier import LazyBinaryClassifierArtifact


class LazyBinaryClassifier(object):
    def __init__(
        self,
        max_heads: int = None,
    ):
        self.is_saved = False
        self.max_heads = max_heads

    def fit(self, X=None, y=None, h5_file=None, h5_idxs=None):
        y = np.array(y, dtype=int)
        self.model = LazyEclecticBinaryClassifier(
            max_heads=self.max_heads,
        )
        self.model.fit(X=X, y=y, h5_file=h5_file, h5_idxs=h5_idxs)

    def predict_proba(self, X=None, h5_file=None, h5_idxs=None):
        y_hat_1 = np.array(self.model.predict(X=X, h5_file=h5_file, h5_idxs=h5_idxs))
        y_hat_0 = 1 - y_hat_1
        return np.array([y_hat_0, y_hat_1]).T

    def predict(self, X=None, h5_file=None, h5_idxs=None, threshold=0.5):
        y_hat = self.predict_proba(X=X, h5_file=h5_file, h5_idxs=h5_idxs)[:, 1]
        return np.where(y_hat >= threshold, 1, 0).astype(int)

    def save_raw(self, model_dir: str):
        self.model.save(model_dir=model_dir)
        self.is_saved = True

    @classmethod
    def load_raw(cls, model_dir: str):
        obj = cls()
        obj.model = LazyEclecticBinaryClassifier.load(model_dir)
        obj.is_saved = True
        return obj

    def save_onnx(self, model_dir: str, clean: bool = True):
        if not self.is_saved:
            self.save(model_dir=model_dir)
        convert_to_onnx(model_dir, clean=clean)

    @classmethod
    def load_onnx(cls, model_dir: str):
        return LazyBinaryClassifierArtifact.load(model_dir=model_dir)

    def save(self, model_dir: str, onnx=True):
        if model_dir.endswith(".zip"):
            zip = True
            model_dir = model_dir[:-4]
        else:
            zip = False
        self.save_raw(model_dir=model_dir)
        if onnx:
            self.save_onnx(model_dir=model_dir, clean=True)
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
        for fn in os.listdir(model_dir):
            if fn.endswith(".onnx"):
                return cls.load_onnx(model_dir=model_dir)
        obj = cls.load_raw(model_dir=model_dir)
        if zip:
            shutil.rmtree(base_dir)
        return obj
