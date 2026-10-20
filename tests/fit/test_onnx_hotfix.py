"""ONNX exports that used to diverge from the model they came from -- pinned so they cannot again.

Each case here reproduced a silent divergence between a fitted estimator and its exported
graph, with nothing in the suite noticing:

- **Applicability domain with NaN.** The numpy scorer median-imputes NaN; the hand-built
  ONNX graph had no imputation step, so NaN propagated into the distance and the score
  collapsed to 0 -- a descriptor vetoed at inference for a molecule it scored normally at
  fit time. A training column that was entirely NaN was also dropped by the imputer,
  leaving the graph expecting fewer features than the descriptor produces.
- **XGBoost regression objectives with a link function.** The export wrapped the booster
  in a default ``XGBRegressor``, and onnxmltools reads the objective from the wrapper, not
  the booster -- so a ``reg:tweedie`` model was exported without its log link.
- **Early stopping with ``num_parallel_tree > 1``.** onnxmltools keeps
  ``best_iteration + 1`` *trees*; the model predicts with that many *rounds*, each of
  ``num_parallel_tree`` trees.
"""

import os

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("xgboost")
pytest.importorskip("onnx")

from lazyqsar.applicability import ApplicabilityDomain, ApplicabilityDomainArtifact  # noqa: E402
from lazyqsar.artifacts.xgboost import XGBoostArtifact  # noqa: E402
from lazyqsar.base.xgboost.model import BaseXGBRegressor  # noqa: E402


# ------------------------------------------------------------------ applicability domain


def _ad_data(seed=0, n=300, p=40):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[rng.random(size=X.shape) < 0.05] = np.nan
    X[:, 7] = np.nan  # a descriptor that is undefined for every training molecule
    return X


@pytest.fixture(scope="module")
def ad_pair(tmp_path_factory):
    X = _ad_data()
    ad = ApplicabilityDomain().fit(X)
    d = str(tmp_path_factory.mktemp("ad"))
    ad.save(d)
    return ad, ApplicabilityDomainArtifact.load(d), X


def test_ad_graph_accepts_the_full_feature_width(ad_pair):
    _, art, X = ad_pair
    assert art.score(X).shape == (X.shape[0],)


@pytest.mark.parametrize("kind", ["clean", "partial_nan", "all_nan"])
def test_ad_artifact_matches_numpy_on_missing_values(ad_pair, kind):
    ad, art, X = ad_pair
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(50, X.shape[1]))
    if kind == "partial_nan":
        Q[rng.random(size=Q.shape) < 0.2] = np.nan
    elif kind == "all_nan":
        Q[:] = np.nan
    np.testing.assert_allclose(art.score(Q), ad.score(Q), rtol=0, atol=1e-5)


# ------------------------------------------------------------------ xgboost regressor


def _fit_xgb(y, X, tmp_path):
    model = BaseXGBRegressor(oof=False).fit(X, y)
    model.save(str(tmp_path))
    return model, XGBoostArtifact.load(str(tmp_path))


@pytest.mark.parametrize(
    "objective, make_y",
    [
        # Positive and skewed: the heuristic chooses Tweedie (log link).
        ("reg:tweedie", lambda r, s: np.exp(1.5 * s + 0.3 * r.normal(size=s.size))),
        # Skewed with both signs: pseudo-Huber.
        ("reg:pseudohubererror", lambda r, s: np.exp(1.5 * s) - 2.0),
        # Symmetric: squared error.
        ("reg:squarederror", lambda r, s: 3.0 * s + 0.3 * r.normal(size=s.size)),
    ],
)
def test_xgb_regressor_export_keeps_the_objective(objective, make_y, tmp_path):
    rng = np.random.default_rng(0)
    # n < 200 takes the heuristic preset directly, which is where the objective is chosen.
    X = rng.normal(size=(150, 12)).astype(np.float32)
    y = make_y(rng, X[:, 0])
    model, art = _fit_xgb(y, X, tmp_path)
    assert model.params_["objective"] == objective
    native = model.predict(X)
    np.testing.assert_allclose(
        art.run(X), native, rtol=0, atol=1e-5 * float(np.abs(native).max())
    )


def test_xgb_export_keeps_every_tree_of_an_early_stopped_parallel_forest(tmp_path):
    import xgboost as xgb

    from lazyqsar.base.xgboost.inspector import inspect

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 10)).astype(np.float32)
    y = X[:, 0] * 2 + rng.normal(size=400)
    dtr = xgb.DMatrix(X[:300], label=y[:300])
    dva = xgb.DMatrix(X[300:], label=y[300:])
    booster = xgb.train(
        {"num_parallel_tree": 3, "max_depth": 4, "eta": 0.3, "verbosity": 0},
        dtr,
        num_boost_round=200,
        evals=[(dva, "val")],
        early_stopping_rounds=5,
        verbose_eval=False,
    )
    assert booster.best_iteration < booster.num_boosted_rounds() - 1

    model = BaseXGBRegressor(oof=False)
    model.booster_ = booster
    model.best_iteration_ = booster.best_iteration
    model.profile_ = inspect(X, y, task="regression")
    model.preset_name_, model.params_, model.portfolio_scores_ = "manual", {}, {}
    model.save(str(tmp_path))
    art = XGBoostArtifact.load(str(tmp_path))
    np.testing.assert_allclose(art.run(X), model.predict(X), rtol=0, atol=1e-5)
    assert os.path.isfile(tmp_path / "xgboost.onnx")


# ------------------------------------------------------------------ the export self-check


def test_export_self_check_refuses_a_graph_that_diverges(tmp_path, monkeypatch):
    """Reintroduce the original Tweedie bug and the save must fail, not write a wrong model.

    This is the guarantee the self-check exists for: a conversion error is caught at
    export time, by the export, regardless of whether any parity test covers the regime.
    """
    import xgboost as xgb

    import lazyqsar.base.xgboost.model as xgb_model
    from lazyqsar.utils.onnx_check import OnnxExportMismatch

    def objective_blind_wrapper(booster, task):
        wrapper = xgb.XGBRegressor()  # the pre-3.8.1 behaviour: default objective
        wrapper._Booster = booster
        return wrapper

    monkeypatch.setattr(
        xgb_model, "_booster_to_sklearn_wrapper", objective_blind_wrapper
    )
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 12)).astype(np.float32)
    y = np.exp(1.5 * X[:, 0] + 0.3 * rng.normal(size=150))
    model = BaseXGBRegressor(oof=False).fit(X, y)
    assert model.params_["objective"] == "reg:tweedie"
    with pytest.raises(OnnxExportMismatch, match="does not reproduce"):
        model.save(str(tmp_path))


def test_ad_self_check_covers_missing_values(tmp_path, monkeypatch):
    """An AD graph without a working imputation step is refused at export.

    Only the *graph* is broken here: its medians are NaN, which reproduces the original
    graph's behaviour (NaN propagates into the distance). The numpy scorer the check
    compares against keeps the real medians.
    """
    from lazyqsar.applicability import domain
    from lazyqsar.utils.onnx_check import OnnxExportMismatch

    ad = ApplicabilityDomain().fit(_ad_data())
    real_medians = ad.imputer_.statistics_.copy()
    real_score = domain.ApplicabilityDomain.score

    def score_with_real_medians(self, X):
        broken = self.imputer_.statistics_
        self.imputer_.statistics_ = real_medians
        try:
            return real_score(self, X)
        finally:
            self.imputer_.statistics_ = broken

    monkeypatch.setattr(domain.ApplicabilityDomain, "score", score_with_real_medians)
    ad.imputer_.statistics_ = np.full_like(real_medians, np.nan)
    with pytest.raises(OnnxExportMismatch):
        ad.save(str(tmp_path))
