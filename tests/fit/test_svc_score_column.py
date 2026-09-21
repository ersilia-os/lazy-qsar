"""The SVC score column must resolve for BOTH skl2onnx converters, canary or not.

`_to_onnx` exports LinearSVC with `raw_scores=True` and kernel SVC with `zipmap=False`.
The two converters order the mutually-negated score columns differently, so any hardcoded
column is right for one flavour and silently inverts the other. lazyqsar <= 3.4.2 hardcoded
column 1 (inverting kernel SVC); 3.4.3's legacy fallback hardcoded column 0 (inverting
LinearSVC).

These tests drive the estimator directly rather than through the portfolio, because which
flavour the portfolio selects depends on the data and cannot be relied on to cover both.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")

import onnxruntime as rt
from sklearn.svm import SVC, LinearSVC

from lazyqsar.base.svc.model import (
    _SCORE_CANARY_ROWS,
    _SCORE_CANARY_SEED,
    BaseSVCArtifact,
    _decision_scores,
    _to_onnx,
)

N, D = 300, 20


def _data(prevalence, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D))
    lg = X @ rng.standard_normal(D)
    y = (lg > np.quantile(lg, 1 - prevalence)).astype(int)
    return X, y


def _artifact(est, tmp_path, with_canary):
    path = str(tmp_path / "svc.onnx")
    _to_onnx(est, path, D)
    meta = {"n_features_in": D, "use_linear": isinstance(est, LinearSVC)}
    if with_canary:
        probe = np.random.default_rng(_SCORE_CANARY_SEED).standard_normal(
            (_SCORE_CANARY_ROWS, D)
        )
        meta["score_canary"] = {
            "seed": _SCORE_CANARY_SEED,
            "n_rows": _SCORE_CANARY_ROWS,
            "decision": _decision_scores(est, probe).tolist(),
        }
    art = BaseSVCArtifact()
    art.metadata = meta
    art._session = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
    art._input_name = art._session.get_inputs()[0].name
    art._score_col = art._resolve_score_column()
    return art


@pytest.mark.parametrize("with_canary", [True, False], ids=["canary", "legacy"])
@pytest.mark.parametrize("prevalence", [0.5, 0.15])
@pytest.mark.parametrize(
    "make_est",
    [
        pytest.param(lambda: LinearSVC(C=1.0, max_iter=5000), id="LinearSVC"),
        pytest.param(lambda: SVC(kernel="rbf", C=1.0), id="SVC-rbf"),
        pytest.param(lambda: SVC(kernel="linear", C=1.0), id="SVC-linear-kernel"),
    ],
)
def test_raw_scores_match_sklearn_decision_function(
    make_est, prevalence, with_canary, tmp_path
):
    """The artifact must return sklearn's decision_function, not its negation.

    `with_canary=False` is the case that matters for every model saved before 3.4.3:
    none of them carry a canary, and a hardcoded fallback inverts half of them.
    """
    X, y = _data(prevalence)
    est = make_est().fit(X, y)
    art = _artifact(est, tmp_path, with_canary)

    expected = _decision_scores(est, X)
    got = art._raw_scores(np.asarray(X, dtype=np.float32))

    # A sign error is the failure this guards; the tolerance only has to exclude it.
    assert np.corrcoef(got, expected)[0, 1] > 0.99, (
        f"score column inverted: correlation with decision_function is "
        f"{np.corrcoef(got, expected)[0, 1]:.3f}"
    )
    assert np.abs(got - expected).max() < 1e-3 * max(1.0, np.abs(expected).max())
