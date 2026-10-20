"""On every committed fixture, the saved checkpoint reproduces the fitted model.

Real molecules, real featurizer (Morgan, fast mode), both tasks. The fitted model's heads
run their own ONNX graphs, so the remaining differences come from two documented places
only: onnxruntime's thread-count-dependent summation order inside tree ensembles, and the
float32 channel accumulation of the deployed ensemble path -- both ~1e-7 to 1e-6.
"""

import contextlib
import io

import numpy as np
import pytest

from _helpers.smiles import (
    load_imbalanced_dataset,
    load_reference_dataset,
    load_regression_dataset,
)

CASES = {
    "binary": (load_reference_dataset, "classification"),
    "imbalanced": (load_imbalanced_dataset, "classification"),
    "regression": (load_regression_dataset, "regression"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_checkpoint_reproduces_the_fitted_model(name, tmp_path):
    from lazyqsar.qsar import LazyQSAR

    loader, task = CASES[name]
    smiles, y = loader()
    with contextlib.redirect_stdout(io.StringIO()):
        model = LazyQSAR(task=task, mode="fast")
        model.fit(smiles, y)
        model.save(str(tmp_path))
        loaded = LazyQSAR.load(str(tmp_path))
        if task == "classification":
            a = model.predict_proba(smiles)[:, 1]
            b = loaded.predict_proba(smiles)[:, 1]
            assert np.array_equal(model.predict(smiles), loaded.predict(smiles))
            atol = 1e-5
        else:
            a, b = model.predict(smiles), loaded.predict(smiles)
            atol = 1e-5 * float(np.std(y))
    np.testing.assert_allclose(b, a, rtol=0, atol=atol)
