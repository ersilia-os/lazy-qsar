"""``allow_kernel_svm`` end to end, on real molecules in fast mode.

With the flag off, a saved SMILES-level checkpoint has no SVM graph and stores no
training row anywhere; with it on, a small dataset gets the SVM head back. The flag is
recorded in the task-level ``metadata.json`` and survives a load.
"""

import contextlib
import io
import json

import numpy as np
import pytest

pytest.importorskip("onnx")

from _helpers.onnx_rows import head_files, training_rows_in_onnx  # noqa: E402
from _helpers.smiles import load_reference_dataset, load_regression_dataset  # noqa: E402

N = 150  # below 300 rows, so both selectors would include the SVM head


def _data(task):
    if task == "classification":
        smiles, y = load_reference_dataset()
        y = np.asarray(y)
        idx = np.r_[np.flatnonzero(y == 1)[:40], np.flatnonzero(y == 0)[: N - 40]]
        return [smiles[i] for i in idx], y[idx]
    smiles, y = load_regression_dataset()
    return list(smiles[:N]), np.asarray(y[:N], dtype=float)


def _fit(task, allow, directory):
    from lazyqsar.qsar import LazyQSAR

    smiles, y = _data(task)
    with contextlib.redirect_stdout(io.StringIO()):
        model = LazyQSAR(task=task, mode="fast", allow_kernel_svm=allow)
        model.fit(smiles, y)
        model.save(str(directory))
    # The training rows each head saw: every descriptor's matrix through each batch's
    # preprocessor.
    rows = []
    for i, agnostic in enumerate(model.models):
        X = model.descriptors[i].transform(smiles)
        rows += [batch.prep.transform(X) for batch in agnostic._model.models]
    return model, np.vstack(rows)


@pytest.fixture(scope="module", params=["classification", "regression"])
def fitted(request, tmp_path_factory):
    task = request.param
    root = tmp_path_factory.mktemp(f"aks-{task}")
    off, rows_off = _fit(task, False, root / "off")
    on, _ = _fit(task, True, root / "on")
    return task, root, off, rows_off, on


def test_denied_checkpoint_has_no_svm_and_no_training_rows(fitted):
    task, root, off, rows_off, _ = fitted
    svm = "svc" if task == "classification" else "svr"
    assert all(svm not in m._model.portfolio for m in off.models)
    assert f"{svm}.onnx" not in head_files(root / "off")
    assert training_rows_in_onnx(root / "off", rows_off) == 0


def test_allowed_checkpoint_has_the_svm_head(fitted):
    task, root, _, _, on = fitted
    svm = "svc" if task == "classification" else "svr"
    assert all(svm in m._model.portfolio for m in on.models)
    assert f"{svm}.onnx" in head_files(root / "on")


def test_the_flag_is_recorded_and_survives_a_load(fitted):
    from lazyqsar.qsar import LazyQSAR

    _, root, _, _, _ = fitted
    for name, expected in (("off", False), ("on", True)):
        meta = json.loads((root / name / "metadata.json").read_text())
        assert meta["allow_kernel_svm"] is expected
    smiles, _ = _data(fitted[0])
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = LazyQSAR.load(str(root / "off"))
        if fitted[0] == "classification":
            a = fitted[2].predict_proba(smiles[:20])
            b = loaded.predict_proba(smiles[:20])
        else:
            a, b = fitted[2].predict(smiles[:20]), loaded.predict(smiles[:20])
    np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-5)
