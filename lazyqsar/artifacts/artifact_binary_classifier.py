import os
import h5py
import shutil
import numpy as np
import onnxruntime as ort


class LazyBinaryClassifierArtifact(object):
    def __init__(self, sessions=None):
        self.sessions = sessions

    def predict_proba(self, X=None, h5_file=None, h5_idxs=None):
        if X is not None:
            X = X.astype("float32")
            if h5_file is not None or h5_idxs is not None:
                raise ValueError("If X is given, h5_file and h5_idxs must be None.")
        else:
            if h5_file is None:
                raise ValueError("If X is None, h5_file must be given.")
            with h5py.File(h5_file, "r") as f:
                keys = list(f.keys())
                if "X" in keys:
                    key = "X"
                elif "data" in keys:
                    key = "data"
                elif "values" in keys:
                    key = "values"
                elif "Values" in keys:
                    key = "Values"
                else:
                    raise ValueError("No dataset found in h5 file.")
                if h5_idxs is None:
                    X = f[key][:].astype("float32")
                else:
                    X = f[key][h5_idxs].astype("float32")
        if self.sessions is None:
            raise ValueError("Model not loaded. Call `load` first.")
        R = []
        for session in self.sessions:
            inputs = {session.get_inputs()[0].name: X}
            R += [session.run(None, inputs)[0].tolist()]
        R = np.array(R)
        y_hat_1 = np.mean(R, axis=0)
        y_hat_0 = 1 - y_hat_1
        return np.array([y_hat_0, y_hat_1]).T

    def predict(self, X=None, h5_file=None, h5_idxs=None, threshold=0.5):
        y_hat = self.predict_proba(X=X, h5_file=h5_file, h5_idxs=h5_idxs)[:, 1]
        y_bin = []
        for y in y_hat:
            if y >= threshold:
                y_bin.append(1)
            else:
                y_bin.append(0)
        return np.array(y_bin, dtype=int)

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
        onnx_files = []
        for fn in os.listdir(model_dir):
            if fn.endswith(".onnx"):
                onnx_files += [os.path.join(model_dir, fn)]
        sessions = []
        for onnx_file in onnx_files:
            sessions += [ort.InferenceSession(onnx_file)]
        obj = cls(sessions=sessions)
        return obj
