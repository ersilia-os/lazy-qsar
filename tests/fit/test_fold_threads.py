"""OOF folds run in threads only where the solver is deterministic under threads.

liblinear (LinearSVC, LinearSVR) shuffles with the process-wide C ``rand()``; fold fits
running concurrently interleave it and the out-of-fold predictions -- and with them the
calibrator, cutoffs and ranks -- change from run to run (measured at ~1e-3 on
TDC lipophilicity). libsvm's kernel solver draws no random numbers.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from lazyqsar.base.svc._common import _fold_jobs  # noqa: E402


def test_liblinear_folds_stay_serial():
    assert _fold_jobs(5, {"use_linear": True}) == 1


def test_kernel_folds_may_thread(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    assert _fold_jobs(5, {"use_linear": False}) == 5
    assert _fold_jobs(3, {}) == 3


def test_threaded_kernel_svr_is_reproducible():
    from lazyqsar.base.svc import BaseSVR

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 20))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.normal(size=300)
    a = BaseSVR().fit(X, y).oof_preds_
    b = BaseSVR().fit(X, y).oof_preds_
    np.testing.assert_array_equal(a, b)
