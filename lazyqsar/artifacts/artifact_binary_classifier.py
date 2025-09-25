import onnxruntime as ort


class LazyBinaryClassifierArtifact(object):

    def __init__(self):
        self.session = None

    def predict(self, X):
        X = X.astype('float32')
        if self.session is None:
            raise ValueError("Model not loaded. Call `load` first.")
        inputs = {self.session.get_inputs()[0].name: X}
        return self.session.run(None, inputs)[0]

    @classmethod
    def load(cls, onnx_file: str):
        obj = cls()
        obj.session = ort.InferenceSession(onnx_file)
        return obj
