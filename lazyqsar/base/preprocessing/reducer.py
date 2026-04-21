"""
Feature selection for base preprocessing.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection._base import SelectorMixin
from sklearn.pipeline import Pipeline

from .inspector import PreprocessingProfile


class CorrelationFilter(BaseEstimator, SelectorMixin):
    """
    Unsupervised feature selector that removes highly correlated features.
    """

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        corr = np.corrcoef(X.T)
        variances = X.var(axis=0)
        mask = np.ones(p, dtype=bool)
        for i in range(p):
            if not mask[i]:
                continue
            for j in range(i + 1, p):
                if not mask[j]:
                    continue
                if abs(corr[i, j]) > self.threshold:
                    if variances[i] >= variances[j]:
                        mask[j] = False
                    else:
                        mask[i] = False
                        break
        self.mask_ = mask
        return self

    def _get_support_mask(self):
        return self.mask_


def select_reducer(profile: PreprocessingProfile) -> str:
    p = profile.n_features
    ratio = profile.n_p_ratio
    if p <= 50 or ratio >= 20:
        return "variance_threshold"
    return "correlation_filter"


def build_reducer(reducer_name: str, profile: PreprocessingProfile):
    if reducer_name == "variance_threshold":
        return VarianceThreshold(threshold=1e-6)
    if reducer_name == "correlation_filter":
        return Pipeline(
            [
                ("vt", VarianceThreshold(threshold=1e-6)),
                ("select", CorrelationFilter(threshold=0.90)),
            ]
        )
    raise ValueError(
        f"Unknown reducer {reducer_name!r}. "
        f"Valid options: variance_threshold, correlation_filter"
    )


def _register_correlation_filter_onnx_converter():
    from skl2onnx import update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.proto import onnx_proto

    def _shape_calc(operator):
        op = operator.raw_operator
        N = operator.inputs[0].type.shape[0]
        n_out = int(op.mask_.sum())
        operator.outputs[0].type = FloatTensorType([N, n_out])

    def _converter(scope, operator, container):
        op = operator.raw_operator
        input_name = operator.inputs[0].full_name
        output_name = operator.outputs[0].full_name
        indices = np.where(op.mask_)[0].astype(np.int64)
        indices_name = scope.get_unique_variable_name("cf_indices")
        container.add_initializer(
            indices_name,
            onnx_proto.TensorProto.INT64,
            [len(indices)],
            indices.tolist(),
        )
        container.add_node(
            "Gather",
            [input_name, indices_name],
            [output_name],
            axis=1,
            name=scope.get_unique_operator_name("GatherCorrelationFilter"),
        )

    update_registered_converter(
        CorrelationFilter,
        "SklearnCorrelationFilter",
        _shape_calc,
        _converter,
        overwrite=True,
    )
