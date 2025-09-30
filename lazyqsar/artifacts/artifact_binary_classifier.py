import os
import numpy as np
import onnxruntime as ort


class LazyBinaryClassifierArtifact(object):

    def __init__(self, sessions=None):
        self.sessions = sessions

    def predict(self, X):
        X = X.astype('float32')
        if self.sessions is None:
            raise ValueError("Model not loaded. Call `load` first.")
        R = []
        for session in self.sessions:
            inputs = {session.get_inputs()[0].name: X}
            R += [session.run(None, inputs)[0].tolist()]
        R = np.array(R)
        y_pred = np.mean(R, axis=0)
        return y_pred

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
