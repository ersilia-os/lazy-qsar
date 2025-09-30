import os
import numpy as np
import onnxruntime as ort


class LazyBinaryClassifierArtifact(object):

    def __init__(self, sessions=None):
        self.sessions = sessions

    def predict_proba(self, X):
        X = X.astype('float32')
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

    def predict(self, X, threshold=0.5):
        y_hat = self.predict_proba(X)[:, 1]
        y_bin = []
        for y in y_hat:
            if y >= threshold:
                y_bin.append(1)
            else:
                y_bin.append(0)
        return np.array(y_bin, dtype=int)

    @classmethod
    def load(cls, model_dir: str):
        onnx_files = []
        for fn in os.listdir(model_dir):
            if fn.endswith(".onnx"):
                onnx_files += [os.path.join(model_dir, fn)]
        sessions = []
        for onnx_file in onnx_files:
            sessions += [ort.InferenceSession(onnx_file)]
        obj = cls(sessions=sessions)
        return obj
