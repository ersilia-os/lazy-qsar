"""Task-neutral name for the predict entry point (classification and regression).

``lazyqsar.api.classifier_predict`` keeps working; it predates regression and is where the
implementation lives.
"""

from .classifier_predict import predict, read_output_array, read_smiles  # noqa: F401

__all__ = ["predict", "read_output_array", "read_smiles"]
