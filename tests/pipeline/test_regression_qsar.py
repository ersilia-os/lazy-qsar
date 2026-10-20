"""``LazyRegressorQSAR`` end to end over stubbed descriptors.

Five stub descriptors (the slow-mode names) are registered; the target is a function of
the ``rdkit`` stub's features only, so exactly one descriptor carries signal and the rest
are noise. That makes the ensemble's choices checkable:

- descriptor screening and pruning must keep the informative descriptor and must not
  leave a noise descriptor active with a meaningful weight,
- the fitted model and its ONNX checkpoint must predict the same values,
- the checkpoint must say ``"task": "regression"`` and carry no classification keys,
- ``LazyQSAR.load`` dispatches on that key, and the classifier loader refuses it.
"""

import contextlib
import io
import json
import os

import numpy as np
import pytest

from _helpers.stubs import install_stub_registry

SLOW = ("cddd", "chemeleon", "clamp", "morgan", "rdkit")
N = 220
_DATA = os.path.join(
    os.path.dirname(__file__), "..", "data", "reference_imbalanced.csv"
)


def _smiles(n):
    """Real ChEMBL strings; with stubbed descriptors only their distinctness matters."""
    with open(_DATA) as f:
        rows = [line.split(",")[0] for line in f.read().splitlines()[1:]]
    return np.array(rows[:n], dtype=object)


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    from lazyqsar.qsar import LazyQSAR

    mp = pytest.MonkeyPatch()
    try:
        install_stub_registry(mp)(*SLOW)
        from lazyqsar.registry import get_descriptor_type

        smiles = _smiles(N)
        F = get_descriptor_type("rdkit")().transform(list(smiles))
        rng = np.random.default_rng(0)
        y = 5.0 + 2.0 * F[:, 0] - F[:, 1] + 0.5 * F[:, 2] + 0.2 * rng.normal(size=N)

        train, test = slice(0, 180), slice(180, N)
        model = LazyQSAR(task="regression", mode="slow")
        _quiet(model.fit, list(smiles[train]), y[train])
        root = str(tmp_path_factory.mktemp("regqsar"))
        _quiet(model.save, root)
        loaded = _quiet(LazyQSAR.load, root)
        yield {
            "model": model,
            "loaded": loaded,
            "root": root,
            "smiles": smiles,
            "y": y,
            "train": train,
            "test": test,
        }
    finally:
        mp.undo()


def test_informative_descriptor_is_kept_and_leads(fitted):
    m = fitted["model"]
    assert "rdkit" in m.descriptor_types
    i = m.descriptor_types.index("rdkit")
    assert m.active_descriptors_[i]
    assert m.oof_rhos_[i] == max(r for r in m.oof_rhos_ if r is not None)
    assert m.oof_rhos_[i] > 0.6


def test_noise_descriptors_are_pruned(fitted):
    m = fitted["model"]
    for name, rho, active in zip(
        m.descriptor_types, m.oof_rhos_, m.active_descriptors_
    ):
        if name != "rdkit":
            assert not active, f"noise descriptor {name} (rho={rho}) left active"


def test_predicts_held_out_compounds(fitted):
    m, smiles, y, test = fitted["model"], fitted["smiles"], fitted["y"], fitted["test"]
    pred = _quiet(m.predict, list(smiles[test]))
    assert pred.shape == (N - 180,)
    from lazyqsar.utils.metrics import spearman_score

    assert spearman_score(y[test], pred) > 0.6
    assert m.oof_metrics_["spearman"] > 0.6


def test_onnx_checkpoint_matches_fitted_model(fitted):
    m, w, smiles = fitted["model"], fitted["loaded"], fitted["smiles"]
    a = _quiet(m.predict, list(smiles))
    b = _quiet(w.predict, list(smiles))
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-4 * fitted["y"].std())


def test_metadata_is_regression_only(fitted):
    with open(os.path.join(fitted["root"], "metadata.json")) as f:
        meta = json.load(f)
    assert meta["task"] == "regression"
    for key in ("oof_rhos", "quality_rhos", "active_descriptors", "rank_error_curves"):
        assert key in meta
    for key in (
        "population_prior",
        "oof_aucs",
        "pooled_ranker",
        "decision_cutoff_proba",
    ):
        assert key not in meta
    for r_knots, e_knots in meta["rank_error_curves"].values():
        assert len(r_knots) == len(e_knots)
        assert all(0.0 <= e <= 1.0 for e in e_knots)


def test_loaded_wrapper_only_opens_active_descriptors(fitted):
    # By name: the loader orders descriptors as the directory lists them (sorted), the
    # fitted model in screening order.
    w, m = fitted["loaded"], fitted["model"]
    opened = {n: a is not None for n, a in zip(w.descriptor_types, w.artifacts)}
    expected = dict(zip(m.descriptor_types, m.active_descriptors_))
    assert opened == expected


def test_empty_query(fitted):
    assert _quiet(fitted["loaded"].predict, []).shape == (0,)


def test_classifier_loader_refuses_regression_checkpoint(fitted):
    from lazyqsar.qsar import LazyClassifierQSAR

    with pytest.raises(ValueError, match="regression checkpoint"):
        LazyClassifierQSAR.load(fitted["root"])
