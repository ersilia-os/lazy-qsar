"""``allow_kernel_svm``: the SVM head is fitted only when allowed.

A kernel SVM's ONNX graph stores its support vectors, which are training rows, so a
checkpoint that contains one exposes part of its training data. With the flag off, the
head selector drops the SVM head (``svc`` or ``svr``) and nothing else changes: the other
heads are chosen exactly as before.
"""

import contextlib
import io
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("onnx")

from _helpers.onnx_rows import head_files, training_rows_in_onnx  # noqa: E402
from lazyqsar.portfolios._base import HeadSelector  # noqa: E402
from lazyqsar.portfolios.classification import Portfolio as ClassifierPortfolio  # noqa: E402
from lazyqsar.portfolios.regression import Portfolio as RegressorPortfolio  # noqa: E402


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def _classification_data(n=150, p=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = (X[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(int)
    return X, y


def _regression_data(n=150, p=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    return X, X[:, 0] * 2 + rng.standard_normal(n)


# ------------------------------------------------------------------ head selection


def test_classification_hard_guard_drops_only_the_svm():
    X, y = _classification_data()  # n < 300: the hard guard includes every head
    allowed, denied = ClassifierPortfolio(True), ClassifierPortfolio(False)
    _quiet(allowed.fit, X, y)
    _quiet(denied.fit, X, y)
    assert allowed.get() == ["lr", "xgb", "rf", "svc"]
    assert denied.get() == ["lr", "xgb", "rf"]
    assert any("allow_kernel_svm=False" in r for r in denied.selector_reasons_)
    assert not any("allow_kernel_svm" in r for r in allowed.selector_reasons_)


def test_classification_scored_path_drops_only_the_svm(monkeypatch):
    X, y = _classification_data(n=1200, p=20)  # no hard guard applies

    def svm_wins(self, profile):
        return {"lr": 3, "xgb": 1, "svc": 3}, []

    monkeypatch.setattr(ClassifierPortfolio, "_score_profile", svm_wins)
    allowed, denied = ClassifierPortfolio(True), ClassifierPortfolio(False)
    _quiet(allowed.fit, X, y)
    _quiet(denied.fit, X, y)
    assert allowed.get() == ["lr", "xgb", "rf", "svc"]
    assert denied.get() == ["lr", "xgb", "rf"]
    # The scores still say what would have been chosen.
    assert denied.selector_scores_["svc"] == 3


@pytest.mark.parametrize(
    "n, allowed, denied",
    [
        (150, ["lr", "xgb", "svr"], ["lr", "xgb"]),
        (2000, ["lr", "xgb", "svr"], ["lr", "xgb"]),
        (3000, ["lr", "xgb"], ["lr", "xgb"]),
        (8000, ["xgb", "rf"], ["xgb", "rf"]),
    ],
)
def test_regression_selector_drops_only_the_svr(n, allowed, denied):
    X, y = _regression_data(n=n)
    on, off = RegressorPortfolio(True), RegressorPortfolio(False)
    _quiet(on.fit, X, y)
    _quiet(off.fit, X, y)
    assert on.get() == allowed
    assert off.get() == denied


@pytest.mark.parametrize("allow", [True, False])
def test_portfolio_json_round_trips_the_flag(tmp_path, allow):
    X, y = _classification_data()
    pf = ClassifierPortfolio(allow)
    _quiet(pf.fit, X, y)
    pf.save(str(tmp_path))
    assert (
        json.loads((tmp_path / "portfolio.json").read_text())["allow_kernel_svm"]
        is allow
    )
    assert HeadSelector.load(str(tmp_path)).allow_kernel_svm is allow


def test_portfolio_json_without_the_flag_loads_as_allowed(tmp_path):
    X, y = _classification_data()
    pf = ClassifierPortfolio(True)
    _quiet(pf.fit, X, y)
    pf.save(str(tmp_path))
    payload = json.loads((tmp_path / "portfolio.json").read_text())
    del payload["allow_kernel_svm"]  # a save from before 3.10
    (tmp_path / "portfolio.json").write_text(json.dumps(payload))
    loaded = HeadSelector.load(str(tmp_path))
    assert loaded.allow_kernel_svm is True
    assert loaded.portfolio == ["lr", "xgb", "rf", "svc"]


# ------------------------------------------------------------------ checkpoints


def _fit_and_save(cls, X, y, allow, directory):
    model = cls(allow_kernel_svm=allow)
    _quiet(model.fit, X=X, y=y)
    _quiet(model.save, str(directory))
    return model


def _preprocessed(model, X):
    return np.vstack([batch.prep.transform(X) for batch in model._model.models])


@pytest.mark.parametrize(
    "task, svm_file",
    [("classification", "svc.onnx"), ("regression", "svr.onnx")],
)
def test_checkpoint_stores_training_rows_only_when_allowed(tmp_path, task, svm_file):
    from lazyqsar.agnostic import LazyClassifier, LazyRegressor

    if task == "classification":
        cls, (X, y) = LazyClassifier, _classification_data()
    else:
        cls, (X, y) = LazyRegressor, _regression_data()

    on = _fit_and_save(cls, X, y, True, tmp_path / "on")
    off = _fit_and_save(cls, X, y, False, tmp_path / "off")

    # The detector works: the allowed checkpoint carries a kernel SVM and its rows.
    assert svm_file in head_files(tmp_path / "on")
    assert training_rows_in_onnx(tmp_path / "on", _preprocessed(on, X)) > 0

    # The default-to-be: no SVM graph, and no training row anywhere.
    assert svm_file not in head_files(tmp_path / "off")
    assert training_rows_in_onnx(tmp_path / "off", _preprocessed(off, X)) == 0

    meta = json.loads((tmp_path / "off" / "metadata.json").read_text())
    assert meta["allow_kernel_svm"] is False
    meta = json.loads((tmp_path / "on" / "metadata.json").read_text())
    assert meta["allow_kernel_svm"] is True


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_denied_checkpoint_round_trips(tmp_path, task):
    from lazyqsar.agnostic import LazyClassifier, LazyRegressor

    if task == "classification":
        cls, (X, y) = LazyClassifier, _classification_data()
    else:
        cls, (X, y) = LazyRegressor, _regression_data()
    model = _fit_and_save(cls, X, y, False, tmp_path)
    loaded = cls.load(str(tmp_path))
    if task == "classification":
        a, b = model.predict_proba(X=X[:20]), loaded.predict_proba(X=X[:20])
    else:
        a, b = model.predict(X=X[:20]), loaded.predict(X=X[:20])
    np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-6)
