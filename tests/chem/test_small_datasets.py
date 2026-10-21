"""Small datasets fit end to end, for both tasks.

The promised minimum is 10 compounds for regression and two of each class for
classification. Since every component is checked against its ONNX export at fit time, a
component that is numerically fragile on few rows -- the applicability domain was -- turns
into a fit that cannot complete. These fits use real molecules and the real featurizer.
"""

import contextlib
import io

import numpy as np
import pytest

from _helpers.smiles import load_reference_dataset, load_regression_dataset


@pytest.mark.parametrize("n", [10, 20, 30, 45])
def test_regression_fits_small_datasets(n, tmp_path):
    from lazyqsar.qsar import LazyQSAR

    smiles, y = load_regression_dataset()
    with contextlib.redirect_stdout(io.StringIO()):
        model = LazyQSAR(task="regression", mode="fast")
        model.fit(smiles[:n], np.asarray(y[:n]))
        model.save(str(tmp_path))
        loaded = LazyQSAR.load(str(tmp_path))
        a, b = model.predict(smiles[n : n + 20]), loaded.predict(smiles[n : n + 20])
    assert np.isfinite(a).all()
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-5 * float(np.std(y[:n])))


@pytest.mark.parametrize("n", [20, 30, 45])
def test_classification_fits_small_datasets(n, tmp_path):
    from lazyqsar.qsar import LazyQSAR

    smiles, y = load_reference_dataset()
    y = np.asarray(y)
    # Keep both classes represented in the first n rows.
    idx = np.r_[np.flatnonzero(y == 1)[: n // 2], np.flatnonzero(y == 0)[: n - n // 2]]
    with contextlib.redirect_stdout(io.StringIO()):
        model = LazyQSAR(task="classification", mode="fast")
        model.fit([smiles[i] for i in idx], y[idx])
        proba = model.predict_proba(smiles[200:220])[:, 1]
    assert np.isfinite(proba).all()
