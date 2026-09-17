"""Read the four per-descriptor prediction channels out of a saved artifact.

One pass over the descriptor matrix produces every channel :func:`combine` might need —
calibrated probability, rank, raw score and applicability-domain score — instead of one
pass per requested output. That matters because featurization dominates inference wall
clock by roughly an order of magnitude, and the caller that re-runs the whole pipeline
once per output type pays it again each time.

Chunked throughout: the matrix stays on disk as a memmap and only ``chunk_size`` rows are
materialised at once, which is what lets a model score a million-compound library.

numpy + onnxruntime only — no RDKit, no scikit-learn. This module sits on the path the
Ersilia Model Hub runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Channels:
    """Per-descriptor predictions for one task, one descriptor.

    Each is ``(n_samples,)`` float32, or ``None`` when that channel was not requested or
    the artifact could not supply it.
    """

    y: np.ndarray | None = None
    r: np.ndarray | None = None
    s: np.ndarray | None = None
    a: np.ndarray | None = None


def required_channels(outputs, has_ad: bool) -> set[str]:
    """Return the channel letters needed to produce *outputs*.

    ``y`` is always needed. ``r`` is needed for the ``rank`` output, and also whenever
    applicability-domain scores are available, because the weighting derives its
    per-sample reliability term from the ranks. ``s`` is needed only for ``score``.

    Skipping a channel is purely an optimisation — it saves ONNX calls and memory — so
    getting this wrong shows up as different numbers, not just slower ones. The runner's
    tests cover every combination.
    """
    want = {"y"}
    if has_ad:
        want.add("a")
        want.add("r")
    if "rank" in outputs:
        want.add("r")
    if "score" in outputs:
        want.add("s")
    return want


def _positive_column(fn, X, label, logger=None):
    """Call *fn(X)* and take the positive-class column, or None if unavailable.

    Mirrors the fallback the non-streaming path has always had: a head that does not
    expose a channel yields ``None`` and the ensemble degrades accordingly. A real failure
    is indistinguishable from that here, so it is logged rather than swallowed silently.
    """
    try:
        return fn(X)[:, 1]
    except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
        if logger is not None:
            logger.warning(
                f"{label} unavailable, falling back: {type(exc).__name__}: {exc}"
            )
        return None


def score_chunkwise(artifact, ad_artifact, x_path, chunk_size, want, logger=None):
    """Score a persisted descriptor matrix, returning the requested channels.

    Parameters
    ----------
    artifact : LazyClassifierArtifact
        Loaded per-descriptor model.
    ad_artifact : ApplicabilityDomainArtifact or None
        Loaded applicability-domain model, when the checkpoint has one.
    x_path : str
        Path to a ``.npy`` written by the descriptor-persistence step.
    chunk_size : int
        Rows per ONNX call.
    want : set of str
        Channel letters, from :func:`required_channels`.
    logger : optional
        Anything with ``.warning``; used only for the fallback above.

    Returns
    -------
    Channels
        float32 vectors of length ``n_samples``. Accumulating in float32 halves the
        memory a multi-task run holds; :func:`combine` upcasts before any arithmetic, so
        the result is unaffected.
    """
    X_mm = np.load(x_path, mmap_mode="r")
    n_total = X_mm.shape[0]

    parts = {k: [] for k in ("y", "r", "s", "a")}
    unavailable = set()

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        # ascontiguousarray materialises just this slice; handing the memmap itself to
        # onnxruntime or the AD artifact would pull the whole matrix into memory.
        X_chunk = np.ascontiguousarray(X_mm[start:end])

        parts["y"].append(np.asarray(artifact.predict_proba(X_chunk))[:, 1])
        if "r" in want:
            r = _positive_column(artifact.predict_rank, X_chunk, "predict_rank", logger)
            if r is None:
                unavailable.add("r")
            else:
                parts["r"].append(r)
        if "s" in want:
            s = _positive_column(
                artifact.predict_score, X_chunk, "predict_score", logger
            )
            if s is None:
                unavailable.add("s")
            else:
                parts["s"].append(s)
        if "a" in want and ad_artifact is not None:
            parts["a"].append(ad_artifact.score(X_chunk))

        del X_chunk

    del X_mm

    def joined(key):
        if key in unavailable or not parts[key]:
            return None
        return np.concatenate(parts[key]).astype(np.float32, copy=False)

    return Channels(y=joined("y"), r=joined("r"), s=joined("s"), a=joined("a"))
