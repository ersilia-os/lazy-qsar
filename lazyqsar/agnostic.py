from .assemblers.default_binary_classifier import LazyDefaultBinaryClassifier, convert_to_onnx
from .artifacts.artifact_binary_classifier import LazyBinaryClassifierArtifact


class LazyBinaryClassifier(object):

    def __init__(self):
        pass

    def fit(self):
        pass

    def save(self, model_dir: str):
        convert_to_onnx(model_dir)

    def load(self, model_dir: str):
        return LazyBinaryClassifierArtifact.load(model_dir: model_dir)
