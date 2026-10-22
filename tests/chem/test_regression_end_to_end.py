"""Regression on real molecules: fast mode (Morgan), RDKit, ONNX round trip.

Two targets. The measured one is EGFR pChEMBL (``reference_regression.csv``). The other,
below, is computed:

The target is Crippen logP computed by RDKit on the committed ChEMBL SMILES, plus noise --
a computed property rather than a measured one, so no bioactivity source beyond ChEMBL is
introduced, and a target Morgan fingerprints can genuinely learn. The tier exists to
exercise the real featurizer, the real SMILES checks and the real export, which the
stubbed pipeline tests cannot.

Needs RDKit.
"""

import contextlib
import io

import numpy as np
import pytest

from _helpers.smiles import load_imbalanced_dataset


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


@pytest.fixture(scope="module")
def data():
    from rdkit import Chem
    from rdkit.Chem import Crippen

    smiles, _ = load_imbalanced_dataset()
    rng = np.random.default_rng(0)
    y = np.array([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in smiles])
    y = y + 0.3 * rng.normal(size=len(y))
    test = np.arange(len(smiles)) % 5 == 0
    return smiles, y, test


@pytest.fixture(scope="module")
def fitted(data, tmp_path_factory):
    from lazyqsar.qsar import LazyQSAR

    smiles, y, test = data
    train_smiles = [s for s, t in zip(smiles, test) if not t]
    model = LazyQSAR(task="regression", mode="fast")
    _quiet(model.fit, train_smiles, y[~test])
    path = _quiet(model.save, str(tmp_path_factory.mktemp("reg") / "model.zip"))
    loaded = _quiet(LazyQSAR.load, path)
    return model, loaded


def test_learns_logp_from_morgan(data, fitted):
    from lazyqsar.utils.metrics import regression_metrics

    smiles, y, test = data
    model, _ = fitted
    test_smiles = [s for s, t in zip(smiles, test) if t]
    m = regression_metrics(y[test], _quiet(model.predict, test_smiles))
    assert m["spearman"] > 0.8
    assert m["r2"] > 0.5
    assert model.oof_metrics_["spearman"] > 0.8


def test_zip_checkpoint_reproduces_predictions(data, fitted):
    smiles, y, _ = data
    model, loaded = fitted
    a = _quiet(model.predict, smiles)
    b = _quiet(loaded.predict, smiles)
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-4 * y.std())


def test_unparseable_smiles_predict_nan(fitted):
    _, loaded = fitted
    out = _quiet(loaded.predict, ["CCO", "not_a_smiles", "c1ccccc1"])
    assert np.isfinite(out[[0, 2]]).all()
    assert np.isnan(out[1])


@pytest.fixture(scope="module")
def egfr():
    from lazyqsar.qsar import LazyQSAR

    from _helpers.smiles import load_regression_dataset

    smiles, y = load_regression_dataset()
    y = np.asarray(y)
    test = np.arange(len(smiles)) % 5 == 0
    model = LazyQSAR(task="regression", mode="fast")
    _quiet(model.fit, [s for s, t in zip(smiles, test) if not t], y[~test])
    return model, smiles, y, test


def test_measured_egfr_potency_is_learned(egfr):
    """Measured pChEMBL, so a realistic bar: Morgan alone ranks held-out EGFR potency.

    ~0.73 Spearman when written; the threshold leaves room for platform variation while
    still failing if the model stopped learning.
    """
    from lazyqsar.utils.metrics import regression_metrics

    model, smiles, y, test = egfr
    pred = _quiet(model.predict, [s for s, t in zip(smiles, test) if t])
    assert regression_metrics(y[test], pred)["spearman"] > 0.55
    assert model.oof_metrics_["spearman"] > 0.55


@pytest.mark.parametrize("kind", ["numpy", "pandas", "with_none"])
def test_predict_accepts_array_like_smiles(fitted, data, kind):
    """numpy arrays and pandas Series predict like lists; a None row comes back NaN."""
    model = fitted[0] if isinstance(fitted, tuple) else fitted
    smiles, _, test = data
    query = [s for s, t in zip(smiles, test) if t][:20]
    want = model.predict(query)
    if kind == "numpy":
        got = model.predict(np.asarray(query, dtype=object))
    elif kind == "pandas":
        pd = pytest.importorskip("pandas")
        got = model.predict(pd.Series(query))
    else:
        got = model.predict(query[:5] + [None] + query[5:])
        assert np.isnan(got[5])
        got = np.delete(got, 5)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_intervals_cover_held_out_molecules(fitted, data):
    """90% split-conformal intervals cover roughly 90% of held-out molecules."""
    model, _ = fitted
    smiles, y, test = data
    query = [s for s, t in zip(smiles, test) if t]
    lower, upper = model.predict_interval(query, level=0.9)
    covered = float(np.mean((y[test] >= lower) & (y[test] <= upper)))
    assert 0.8 <= covered <= 0.99, covered
    assert np.all(upper > lower)


def test_intervals_survive_the_export(fitted, data):
    model, loaded = fitted
    smiles, _, test = data
    query = [s for s, t in zip(smiles, test) if t][:40]
    for level in (0.5, 0.9):
        a = np.column_stack(model.predict_interval(query, level=level))
        b = np.column_stack(loaded.predict_interval(query, level=level))
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_wider_level_gives_wider_interval(fitted, data):
    model, _ = fitted
    smiles, _, test = data
    query = [s for s, t in zip(smiles, test) if t][:40]
    lo50, hi50 = model.predict_interval(query, level=0.5)
    lo95, hi95 = model.predict_interval(query, level=0.95)
    assert np.all(hi95 - lo95 >= hi50 - lo50)
