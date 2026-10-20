"""Regression on real molecules: fast mode (Morgan), RDKit, ONNX round trip.

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
