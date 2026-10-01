"""Does fitting stay inside memory when the preprocessor runs as an ONNX session?

``BasePreprocessor.fit`` binds an onnxruntime session to the fitted pipeline
(``_bind_onnx_runtime``) so the heads are fitted on exactly the values they will be served.
That session stays alive for the whole fit, and a ``LazyClassifier`` holds one per batch per
descriptor: an imbalanced dataset is cut into ``ceil(n_negatives / (100 * n_actives))``
batches, so 19 actives among 121,000 compounds make 65 of them.

Fitting then scores the *whole* training set through every batch (training AUC, the
rank-error curve, the out-of-fold reference). By default onnxruntime keeps what a run
allocated in a per-session arena for reuse, so each session retains buffers sized to the
largest matrix it has ever seen. Memory then grows with ``batches x rows x width`` -- and
because the batch count itself grows with the rows, quadratically with the dataset. Measured
on a real fit this took one 121,000-compound dataset from 15.6 GB (v3.4.2, scikit-learn
transform, freed after every call) to over 128 GB, and a 328,000-compound one past 125 GB.

These tests pin the two halves of the fix: the session must not pool buffers, and turning
pooling off must not change a single output value, since bit-identity is the reason the
session exists.
"""

import contextlib
import gc
import io

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")
psutil = pytest.importorskip("psutil")
rt = pytest.importorskip("onnxruntime")

from lazyqsar.base.preprocessing.pipeline import BasePreprocessor  # noqa: E402

N_FIT, N_BIG, P = 1500, 25_000, 256


def _fit_preprocessor(seed: int) -> BasePreprocessor:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_FIT, P)).astype("float32")
    y = (rng.random(N_FIT) < 0.3).astype(int)
    with contextlib.redirect_stdout(io.StringIO()):
        return BasePreprocessor().fit(X, y)


def _rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def test_the_fit_time_session_does_not_pool_buffers():
    prep = _fit_preprocessor(0)
    assert prep._onnx_session_ is not None, (
        "the preprocessor should be bound to its ONNX form"
    )
    options = prep._onnx_session_.get_session_options()
    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False


def test_session_options_do_not_change_a_single_output_value():
    """Bit-identical to a session with onnxruntime's defaults -- the whole point of the session."""
    prep = _fit_preprocessor(1)
    X = np.random.default_rng(2).standard_normal((3000, P)).astype("float32")
    default = rt.InferenceSession(
        prep._to_onnx_bytes(), providers=["CPUExecutionProvider"]
    )
    expected = default.run(None, {default.get_inputs()[0].name: X})[0]
    assert np.array_equal(prep.transform(X), expected)


def test_memory_does_not_scale_with_the_number_of_sessions():
    """Each session may keep a small working set, not a copy of every matrix it has seen.

    Ten preprocessors -- ten "batches" of one descriptor -- each transform the same large
    matrix once, as the fit does when it scores the full training set through every batch.
    With the default arena each one retained ~2.1x the matrix (measured on onnxruntime
    1.20.1); without it, well under 1x. The bound sits between the two.
    """
    k = 10
    preps = [_fit_preprocessor(seed) for seed in range(k)]
    big = np.random.default_rng(99).standard_normal((N_BIG, P)).astype("float32")

    gc.collect()
    before = _rss_bytes()
    for prep in preps:
        out = prep.transform(big)
        del out
    gc.collect()
    retained_per_session = (_rss_bytes() - before) / k

    assert retained_per_session < 1.5 * big.nbytes, (
        f"each session kept {retained_per_session / big.nbytes:.2f}x the {big.nbytes / 1e6:.0f} MB "
        f"matrix it transformed; sessions are pooling buffers again"
    )
