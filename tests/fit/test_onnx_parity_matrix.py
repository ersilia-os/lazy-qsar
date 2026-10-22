"""Every head, every regime: the saved ONNX artifact reproduces the native model.

The other export tests reach whichever regime their data happens to select, which left
most of the matrix unexercised: isotonic calibration, LinearSVC/LinearSVR, the SGD and
high-dimensional linear regimes, every XGBoost preset, the robust and power scalers.
Here each regime is *forced* -- explicit parameters, a patched preset race, a lowered
calibration threshold -- and the artifact loaded from disk is compared with:

1. the **native** model (scikit-learn / xgboost, not ONNX), on the raw output -- this is
   what catches a conversion error, and
2. the **fitted** model's own predictions, which since heads are bound to ONNX run the
   same graph as the artifact.

Query rows mix training rows, fresh draws, and training rows nudged one float32 ULP
either side (which is where a tree split threshold would expose a precision mismatch).

Tolerances: probabilities 1e-5 absolute; continuous outputs 1e-5 x the scale of the
predictions. The ONNX graphs compute in float32 against float64 natively, and
onnxruntime's tree ensembles sum per-thread partial results in a thread-count-dependent
order -- both measured at ~1e-6 or below. A conversion bug moves outputs by tenths.
"""

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("xgboost")
pytest.importorskip("skl2onnx")

from lazyqsar.artifacts.linear import LinearArtifact  # noqa: E402
from lazyqsar.artifacts.rf import RandomForestArtifact  # noqa: E402
from lazyqsar.artifacts.svc import SVCArtifact, SVRArtifact  # noqa: E402
from lazyqsar.artifacts.xgboost import XGBoostArtifact  # noqa: E402

P_ATOL = 1e-5
V_RTOL = 1e-5


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def _dense(n=260, p=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)).astype(np.float32)


def _fingerprint(n=260, p=512, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, p)) < 0.05).astype(np.float32)


def _labels(X, seed=0):
    rng = np.random.default_rng(seed)
    s = X[:, 0] - X[:, 1] + 0.5 * rng.normal(size=len(X))
    return (s > np.median(s)).astype(int)


def _target(X, seed=0):
    rng = np.random.default_rng(seed)
    return 40.0 * (X[:, 0] - 0.5 * X[:, 1]) + 2.0 * rng.normal(size=len(X)) + 300.0


def _queries(X, seed=1):
    """Training rows, fresh rows, and training rows one float32 ULP either side."""
    rng = np.random.default_rng(seed)
    train = X[:40]
    fresh = rng.permutation(X)[:40] + rng.normal(
        scale=0.1, size=(40, X.shape[1])
    ).astype(np.float32)
    up = np.nextafter(train, np.float32(np.inf))
    down = np.nextafter(train, np.float32(-np.inf))
    return np.vstack([train, fresh, up, down]).astype(np.float32)


def _close_values(got, want):
    scale = max(1.0, float(np.abs(want).max()))
    np.testing.assert_allclose(got, want, rtol=0, atol=V_RTOL * scale)


def _roundtrip(model, artifact_cls, tmp_path):
    _quiet(model.save, str(tmp_path))
    return artifact_cls.load(str(tmp_path))


# ------------------------------------------------------------------ classifiers


def _check_classifier(model, artifact, Q):
    # 1. raw output: artifact vs the native model. The SVC's native raw output is its
    #    decision value; its artifact reports sigmoid(decision) as the raw score.
    native = model._raw_native(Q)
    if hasattr(model, "svc_"):
        native = 1.0 / (1.0 + np.exp(-native))
    np.testing.assert_allclose(
        artifact.predict_score(Q)[:, 1], native, rtol=0, atol=P_ATOL
    )
    # 2. calibrated output: artifact vs the fitted (bound) model
    np.testing.assert_allclose(
        artifact.run(Q)[:, 1], model.predict_proba(Q)[:, 1], rtol=0, atol=P_ATOL
    )


@pytest.fixture
def force_xgb_preset(monkeypatch):
    """Replace the XGBoost preset race with a fixed winner."""
    from lazyqsar.base.xgboost import model as xm
    from lazyqsar.base.xgboost import presets

    builders = {
        "heuristic": xm._get_params,
        "default": presets.xgb_default_params,
        "flaml": presets.flaml_params,
        "autogluon": presets.autogluon_params,
    }

    def force(name, rounds=60):
        def fake_select(X, y, profile, device, nthread=-1):
            return name, builders[name](profile, device, nthread=nthread), rounds, {}

        monkeypatch.setattr(xm, "_portfolio_select", fake_select)

    return force


@pytest.mark.parametrize("preset", ["heuristic", "default", "flaml", "autogluon"])
@pytest.mark.parametrize("data", ["dense", "fingerprint"])
def test_xgb_classifier_presets(preset, data, force_xgb_preset, tmp_path):
    from lazyqsar.base.xgboost import BaseXGBClassifier

    force_xgb_preset(preset)
    X = _dense() if data == "dense" else _fingerprint()
    model = _quiet(BaseXGBClassifier().fit, X, _labels(X))
    _check_classifier(model, _roundtrip(model, XGBoostArtifact, tmp_path), _queries(X))


@pytest.mark.parametrize(
    "head",
    ["xgb", "linear", "rf", "svc"],
)
def test_isotonic_calibration_round_trips(head, monkeypatch, tmp_path):
    """Isotonic calibration is otherwise only reached with >= 500 minority samples."""
    import importlib

    module = {
        "xgb": "lazyqsar.base.xgboost.model",
        "linear": "lazyqsar.base.linear.model",
        "rf": "lazyqsar.base.randomforest.model",
        "svc": "lazyqsar.base.svc.model",
    }[head]
    mod = importlib.import_module(module)
    # One threshold for every head now, in the shared head base.
    monkeypatch.setattr(
        importlib.import_module("lazyqsar.base._head"),
        "CALIBRATION_ISOTONIC_MIN_MINORITY",
        10,
    )
    cls = {
        "xgb": mod.__dict__.get("BaseXGBClassifier"),
        "linear": mod.__dict__.get("BaseLinearClassifier"),
        "rf": mod.__dict__.get("BaseRFClassifier"),
        "svc": mod.__dict__.get("BaseSVCClassifier"),
    }[head]
    art_cls = {
        "xgb": XGBoostArtifact,
        "linear": LinearArtifact,
        "rf": RandomForestArtifact,
        "svc": SVCArtifact,
    }[head]
    X = _dense()
    model = _quiet(cls().fit, X, _labels(X))
    assert model.calibrator_method_ == "isotonic"
    _check_classifier(model, _roundtrip(model, art_cls, tmp_path), _queries(X))


@pytest.mark.parametrize(
    "regime, X",
    [
        ("standard", _dense()),
        ("high_dim", _dense(n=120, p=300)),
        ("large", _dense()),
    ],
)
def test_linear_classifier_regimes(regime, X, tmp_path):
    from lazyqsar.base.linear import BaseLinearClassifier

    model = _quiet(BaseLinearClassifier(regime=regime).fit, X, _labels(X))
    assert model.regime_ == regime
    _check_classifier(model, _roundtrip(model, LinearArtifact, tmp_path), _queries(X))


@pytest.mark.parametrize("data", ["dense", "fingerprint"])
@pytest.mark.parametrize("n", [150, 260])  # below and above the OOB portfolio threshold
def test_rf_classifier(data, n, tmp_path):
    from lazyqsar.base.randomforest import BaseRFClassifier

    X = _dense(n=n) if data == "dense" else _fingerprint(n=n)
    model = _quiet(BaseRFClassifier().fit, X, _labels(X))
    _check_classifier(
        model, _roundtrip(model, RandomForestArtifact, tmp_path), _queries(X)
    )


@pytest.mark.parametrize("kind", ["kernel", "linear"])
def test_svc_kernel_and_linear(kind, tmp_path):
    from lazyqsar.base.svc import BaseSVCClassifier

    # Fingerprints select LinearSVC, dense descriptors the RBF kernel.
    X = _fingerprint() if kind == "linear" else _dense()
    model = _quiet(BaseSVCClassifier(portfolio=False).fit, X, _labels(X))
    assert model._use_linear_ == (kind == "linear")
    art = _roundtrip(model, SVCArtifact, tmp_path)
    # The SVC's raw score is sigmoid(decision); compare decision via the sigmoid scale.
    np.testing.assert_allclose(
        art.predict_score(X[:80])[:, 1],
        1.0 / (1.0 + np.exp(-model._raw_native(X[:80]))),
        rtol=0,
        atol=P_ATOL,
    )
    np.testing.assert_allclose(
        art.run(_queries(X))[:, 1],
        model.predict_proba(_queries(X))[:, 1],
        rtol=0,
        atol=P_ATOL,
    )


@pytest.mark.parametrize("kind", ["kernel", "linear"])
def test_svc_graph_score_column_is_the_decision_function(kind, tmp_path):
    """The bound score column is sklearn's decision_function, sign and all.

    The two converters order the negated score columns differently (kernel: column 0,
    LinearSVC: column 1), and reading the wrong one inverts every prediction. Pinned for
    both flavours: the bound column equals decision_function, the other column is its
    negation, the graph declares the (n, 2) it computes, the loaded artifact resolves the
    same column, and the fitted and loaded outputs agree exactly.
    """
    import onnx
    import onnxruntime as rt

    from lazyqsar.base.svc import BaseSVCClassifier

    X = _fingerprint() if kind == "linear" else _dense()
    model = _quiet(BaseSVCClassifier(portfolio=False).fit, X, _labels(X))
    assert model._use_linear_ == (kind == "linear")
    assert model._onnx_column_ == (1 if kind == "linear" else 0)

    Q = _queries(X)
    session = rt.InferenceSession(
        model._onnx_bytes_, providers=["CPUExecutionProvider"]
    )
    labels, scores = session.run(
        None, {session.get_inputs()[0].name: Q.astype(np.float32)}
    )
    decision = model.svc_.decision_function(Q)
    col = model._onnx_column_
    scale = max(1.0, float(np.abs(decision).max()))
    np.testing.assert_allclose(scores[:, col], decision, rtol=0, atol=1e-5 * scale)
    np.testing.assert_allclose(scores[:, 1 - col], -decision, rtol=0, atol=1e-5 * scale)
    np.testing.assert_array_equal(labels.ravel(), model.svc_.predict(Q))
    np.testing.assert_array_equal(model._raw(Q), scores[:, col])

    declared = onnx.load_from_string(model._onnx_bytes_).graph.output[1]
    assert [d.dim_value for d in declared.type.tensor_type.shape.dim][1] == 2
    assert scores.shape == (len(Q), 2)

    art = _roundtrip(model, SVCArtifact, tmp_path)
    assert art._score_col == col
    # The graph's column is pinned exactly above; the calibrated probabilities differ by
    # float64 rounding between the head's and the artifact's calibrator code (1.1e-16
    # on x86 Linux).
    np.testing.assert_allclose(
        art.run(Q)[:, 1], model.predict_proba(Q)[:, 1], rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("data", ["dense", "fingerprint"])
def test_rbf_svr_graph_matches_sklearn_and_skl2onnx(data):
    """The kernel SVR's MatMul graph computes what sklearn and skl2onnx's operator do.

    The RBF graph is LazyQSAR's own (``_rbf_svr_onnx_bytes``), not a converter's: it is
    held to sklearn's prediction directly, and to skl2onnx's ``SVMRegressor`` graph, on
    training rows, fresh rows and ULP-nudged rows.
    """
    import onnx
    import onnxruntime as rt
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.svm import SVR

    from lazyqsar.base.svc.regressor import _svr_onnx_bytes

    X = _fingerprint() if data == "fingerprint" else _dense()
    svr = SVR(C=10.0, epsilon=0.05, gamma="scale").fit(X, _target(X))
    Q = _queries(X).astype(np.float32)
    ours = _svr_onnx_bytes(svr, X.shape[1])
    assert "MatMul" in {n.op_type for n in onnx.load_from_string(ours).graph.node}
    reference = convert_sklearn(
        svr,
        initial_types=[("float_input", FloatTensorType([None, X.shape[1]]))],
        target_opset=15,
    ).SerializeToString()

    def run(graph):
        s = rt.InferenceSession(graph, providers=["CPUExecutionProvider"])
        return s.run(None, {s.get_inputs()[0].name: Q})[0].ravel()

    native = svr.predict(Q.astype(np.float64))
    scale = max(1.0, float(np.abs(native).max()))
    np.testing.assert_allclose(run(ours), native, rtol=0, atol=V_RTOL * scale)
    np.testing.assert_allclose(run(ours), run(reference), rtol=0, atol=V_RTOL * scale)


# ------------------------------------------------------------------ regressors


def _check_regressor(model, artifact, Q, native_scale=None):
    # 1. artifact vs the native model (SVR's native output is standardised)
    native = model._raw_native(Q)
    if native_scale is not None:
        native = native * native_scale[0] + native_scale[1]
    _close_values(artifact.run(Q), native)
    # 2. artifact vs the fitted (bound) model
    _close_values(artifact.run(Q), model.predict(Q))


@pytest.mark.parametrize(
    "n", [40, 260, 2100], ids=["fixed_rounds", "early_stopped", "large_preset"]
)
@pytest.mark.parametrize("data", ["dense", "fingerprint"])
def test_xgb_regressor_branches(n, data, tmp_path):
    """Every branch of the regression preset: fixed rounds below 50 rows, early stopping,
    and the faster preset above 2,000 rows."""
    from lazyqsar.base.xgboost import BaseXGBRegressor

    X = _dense(n=n) if data == "dense" else _fingerprint(n=n)
    model = _quiet(BaseXGBRegressor().fit, X, _target(X))
    expected_depth = 8 if n > 2000 else 11
    assert model.params_["max_depth"] == expected_depth
    _check_regressor(model, _roundtrip(model, XGBoostArtifact, tmp_path), _queries(X))


def test_xgb_regressor_survives_tied_targets(tmp_path):
    """Censored data piles many rows on one value; the early-stopping split must not crash."""
    from lazyqsar.base.xgboost import BaseXGBRegressor

    X = _dense(n=300)
    y = np.full(300, 4.0)
    y[:3] = 3.0
    y[-60:] = _target(X)[-60:]
    model = _quiet(BaseXGBRegressor().fit, X, y)
    _check_regressor(model, _roundtrip(model, XGBoostArtifact, tmp_path), _queries(X))


def test_xgb_handles_missing_values_like_xgboost(tmp_path):
    """XGBoost routes NaN down each split's default branch; the graph must do the same."""
    from lazyqsar.base.xgboost import BaseXGBRegressor

    X = _dense()
    model = _quiet(BaseXGBRegressor(oof=False).fit, X, _target(X))
    art = _roundtrip(model, XGBoostArtifact, tmp_path)
    Q = _queries(X)
    Q[::3, ::4] = np.nan
    _close_values(art.run(Q), model._raw_native(Q))


@pytest.mark.parametrize(
    "regime, X",
    [
        ("standard", _dense()),
        ("high_dim", _dense(n=120, p=300)),
        ("large", _dense()),
    ],
)
def test_linear_regressor_regimes(regime, X, tmp_path):
    from lazyqsar.base.linear import BaseLinearRegressor

    model = _quiet(BaseLinearRegressor(regime=regime).fit, X, _target(X))
    assert model.regime_ == regime
    _check_regressor(model, _roundtrip(model, LinearArtifact, tmp_path), _queries(X))


@pytest.mark.parametrize("oof_source", ["folds", "oob"])
def test_rf_regressor_oob_path(oof_source, tmp_path):
    """OOB predictions as OOF: complete, honest, and the export unchanged."""
    from lazyqsar.base.randomforest import BaseRFRegressor

    X = _dense(n=300)
    y = _target(X)
    model = _quiet(BaseRFRegressor(oof_source=oof_source).fit, X, y)
    assert np.isfinite(model.oof_preds_).all()
    if oof_source == "oob":
        np.testing.assert_array_equal(model.oof_preds_, model.oob_preds_)
    _check_regressor(
        model, _roundtrip(model, RandomForestArtifact, tmp_path), _queries(X)
    )


def test_rf_rows_never_out_of_bag_are_not_passed_off_as_predictions():
    """With few trees some rows are in every bootstrap; they are NaN, and folds are used."""
    from lazyqsar.base.randomforest import BaseRFRegressor

    X = _dense(n=300)
    model = _quiet(BaseRFRegressor(n_estimators=3, oof_source="oob").fit, X, _target(X))
    assert np.isnan(model.oob_preds_).any()
    assert np.isfinite(model.oof_preds_).all()
    assert model.timing_["oof_folds"], "fell back to fold refits"


@pytest.mark.parametrize("data", ["dense", "fingerprint"])
@pytest.mark.parametrize("n", [150, 260])
def test_rf_regressor(data, n, tmp_path):
    from lazyqsar.base.randomforest import BaseRFRegressor

    X = _dense(n=n) if data == "dense" else _fingerprint(n=n)
    model = _quiet(BaseRFRegressor().fit, X, _target(X))
    _check_regressor(
        model, _roundtrip(model, RandomForestArtifact, tmp_path), _queries(X)
    )


@pytest.mark.parametrize("kind", ["kernel", "linear"])
def test_svr_kernel_and_linear(kind, tmp_path, monkeypatch):
    """The kernel preset, and the LinearSVR the ONNX size guard falls back to."""
    from lazyqsar.base.svc import BaseSVR

    if kind == "linear":
        # No kernel model fits a zero-byte budget, so the guard ends on LinearSVR.
        monkeypatch.setattr(BaseSVR, "_onnx_max_bytes", 0)
    X = _fingerprint() if kind == "linear" else _dense()
    model = _quiet(BaseSVR(portfolio=False).fit, X, _target(X))
    assert model._use_linear_ == (kind == "linear")
    _check_regressor(
        model,
        _roundtrip(model, SVRArtifact, tmp_path),
        _queries(X),
        native_scale=(model.y_std_, model.y_mean_),
    )


# ------------------------------------------------------------------ preprocessor


@pytest.mark.parametrize("scaler", ["standard", "robust", "power", "max_abs"])
@pytest.mark.parametrize("reducer", ["variance_threshold", "correlation_filter"])
def test_preprocessor_binds_and_round_trips(scaler, reducer, monkeypatch, tmp_path):
    """Every scaler x reducer binds to its ONNX graph and reloads bit-identically."""
    from lazyqsar.artifacts.preprocessor import PreprocessorArtifact
    from lazyqsar.base.preprocessing import BaseClassifierPreprocessor
    from lazyqsar.base.preprocessing import pipeline

    monkeypatch.setattr(pipeline, "select_scaler", lambda profile: scaler)
    monkeypatch.setattr(pipeline, "select_reducer", lambda profile: reducer)
    X = _dense(p=24)
    X[:, 5] = X[:, 4] * 0.999 + 0.001  # a correlated pair for the filter to act on
    prep = _quiet(BaseClassifierPreprocessor().fit, X, _labels(X))
    assert prep.scaler_name_ == scaler and prep.reducer_name_ == reducer
    assert prep._onnx_session_ is not None, "binding fell back to scikit-learn"
    prep.save(str(tmp_path))
    art = PreprocessorArtifact.load(str(tmp_path))
    Q = _queries(X)
    np.testing.assert_array_equal(art.run(Q), prep.transform(Q))
    np.testing.assert_allclose(
        prep.transform(Q), prep.pipeline_.transform(Q).astype(np.float32), atol=1e-4
    )


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_zero_iqr_column_is_scaled(task, monkeypatch, tmp_path):
    """A column constant on most rows (an RDKit sentinel) must not keep its raw units.

    ``RobustScaler`` divides a zero-IQR column by one; the preprocessor divides it by
    its standard deviation instead, in the bound graph and the checkpoint too.
    """
    from lazyqsar.artifacts.preprocessor import PreprocessorArtifact
    from lazyqsar.base.preprocessing import (
        BaseClassifierPreprocessor,
        BaseRegressorPreprocessor,
        pipeline,
    )

    monkeypatch.setattr(pipeline, "select_scaler", lambda profile: "robust")
    monkeypatch.setattr(
        pipeline, "select_reducer", lambda profile: "correlation_filter"
    )
    X = _dense(p=12)
    X[:, 3] = 1e5
    X[::30, 3] = 0.0  # 1e5 on ~97% of rows, as RDKit writes it
    if task == "regression":
        prep = _quiet(BaseRegressorPreprocessor().fit, X, _target(X))
    else:
        prep = _quiet(BaseClassifierPreprocessor().fit, X, _labels(X))
    assert np.abs(prep.transform(X)).max() < 100
    prep.save(str(tmp_path))
    Q = _queries(X)
    np.testing.assert_array_equal(
        PreprocessorArtifact.load(str(tmp_path)).run(Q), prep.transform(Q)
    )


# ------------------------------------------------------------------ fold models


def test_fold_models_stay_close_to_their_onnx_form():
    """OOF statistics come from native fold models; they must not drift from ONNX.

    The final heads are bound to ONNX, fold models are not (exporting each would
    multiply fit cost). This bounds the difference on a fold-sized model of each type.
    """
    from lazyqsar.base.randomforest import BaseRFRegressor
    from lazyqsar.base.xgboost import BaseXGBRegressor

    X = _dense()
    y = _target(X)
    for cls in (BaseXGBRegressor, BaseRFRegressor):
        model = _quiet(cls(oof=False).fit, X[:200], y[:200])
        Q = X[200:]
        _close_values(model._raw_onnx(Q), model._raw_native(Q))


# ------------------------------------------------------------------ the checks themselves


def test_bind_check_catches_a_single_dropped_tree():
    """The bind-time check must be tight enough to notice one missing tree."""
    import xgboost as xgb

    from lazyqsar.base.xgboost import BaseXGBRegressor
    from lazyqsar.base.xgboost.model import _rounds_used
    from lazyqsar.utils.onnx_check import OnnxExportMismatch

    X = _dense()
    model = _quiet(BaseXGBRegressor(oof=False).fit, X, _target(X))
    full = model.booster_
    # A graph built from one round fewer than the model predicts with.
    model.booster_ = _rounds_used(full, model.best_iteration_ - 1)
    data = model._onnx_bytes()
    model.booster_ = full
    model._onnx_bytes = lambda: data
    with pytest.raises(OnnxExportMismatch):
        model._bind_onnx()
    assert isinstance(full, xgb.Booster)


def test_preprocessor_bind_failure_raises_instead_of_falling_back(monkeypatch):
    from lazyqsar.base.preprocessing import BaseRegressorPreprocessor
    from lazyqsar.base.preprocessing import pipeline
    from lazyqsar.utils.onnx_check import OnnxExportMismatch

    def broken(self):
        raise RuntimeError("converter exploded")

    monkeypatch.setattr(pipeline.BasePreprocessor, "_to_onnx_bytes", broken)
    with pytest.raises(OnnxExportMismatch, match="could not be exported"):
        _quiet(BaseRegressorPreprocessor().fit, _dense(), _target(_dense()))


def test_preprocessor_saves_the_bytes_it_was_bound_to(tmp_path):
    from lazyqsar.base.preprocessing import BaseClassifierPreprocessor

    X = _dense()
    prep = _quiet(BaseClassifierPreprocessor().fit, X, _labels(X))
    prep.to_onnx(str(tmp_path / "p.onnx"))
    assert (tmp_path / "p.onnx").read_bytes() == prep._onnx_bytes_


@pytest.mark.parametrize("head", ["xgb", "rf", "linear", "svr"])
def test_pickled_heads_rebind_and_predict_identically(head):
    """A bound head survives pickling: the session is rebuilt from the same bytes."""
    import pickle

    from lazyqsar.base.linear import BaseLinearRegressor
    from lazyqsar.base.randomforest import BaseRFRegressor
    from lazyqsar.base.svc import BaseSVR
    from lazyqsar.base.xgboost import BaseXGBRegressor

    cls = {
        "xgb": BaseXGBRegressor,
        "rf": BaseRFRegressor,
        "linear": BaseLinearRegressor,
        "svr": BaseSVR,
    }[head]
    X = _dense()
    model = _quiet(cls(oof=False).fit, X, _target(X))
    clone = pickle.loads(pickle.dumps(model))
    Q = _queries(X)
    np.testing.assert_array_equal(clone.predict(Q), model.predict(Q))


def test_svc_artifact_refuses_a_score_column_that_disagrees_with_the_fit(tmp_path):
    """The bound column is saved; a loader resolving the other one must raise, not invert."""
    import json

    from lazyqsar.base.svc import BaseSVCClassifier
    from lazyqsar.utils.onnx_check import OnnxExportMismatch

    X = _dense()
    model = _quiet(BaseSVCClassifier().fit, X, _labels(X))
    _quiet(model.save, str(tmp_path))
    meta_path = tmp_path / "svc.json"
    meta = json.loads(meta_path.read_text())
    assert SVCArtifact.load(str(tmp_path))._score_col == meta["score_column"]
    meta["score_column"] = 1 - meta["score_column"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(OnnxExportMismatch):
        SVCArtifact.load(str(tmp_path))
