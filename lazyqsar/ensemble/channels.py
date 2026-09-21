"""Read the four per-descriptor prediction channels out of a saved artifact.

One pass over the descriptor matrix produces every channel :func:`combine` might need —
calibrated probability, rank, raw score and applicability-domain score — instead of one
pass per requested output, and the rank is derived from the probability rather than asked
for separately. That matters because featurization dominates inference wall
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


def required_channels(outputs, has_ad: bool, has_scorer: bool = False) -> set[str]:
    """Return the channel letters needed to produce *outputs*.

    ``y`` is always needed. ``r`` is needed for the ``rank`` output, and also whenever
    applicability-domain scores are available, because the weighting derives its
    per-sample reliability term from the ranks.

    ``s`` is needed only for ``score``, and only on a checkpoint with no pooled score map.
    With one, ``score`` is read off the pooled probability, so the raw channel is never
    looked at -- which is what takes ``predict_type="score"`` from two passes over every
    graph down to one, the same cost as every other output.

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
    if "score" in outputs and not has_scorer:
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


def _score_chunks(chunks, artifact, ad_artifact, want, logger=None):
    """Run every requested channel over an iterable of descriptor-matrix chunks.

    Shared by both channel sources: one reads a persisted matrix off disk, the other
    featurizes on the fly. Neither ever holds more than one chunk.
    """
    parts = {k: [] for k in ("y", "r", "s", "a")}
    unavailable = set()

    for X_chunk in chunks:
        y = np.asarray(artifact.predict_proba(X_chunk))[:, 1]
        parts["y"].append(y)
        if "r" in want:
            # On a pooled-reference checkpoint the rank is a lookup against the training
            # distribution, so it comes out of the probability already computed a line
            # above. Asking `predict_rank` instead would re-run every preprocessor and
            # every head to reach the identical number -- roughly doubling the ONNX work
            # of the two most common requests, since `rank` is the deployed default and an
            # applicability domain forces ranks on even for plain `proba`.
            from_proba = getattr(artifact, "rank_from_proba", None)
            r = from_proba(y) if from_proba is not None else None
            if r is None:
                r = _positive_column(
                    artifact.predict_rank, X_chunk, "predict_rank", logger
                )
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

    def joined(key):
        if key in unavailable or not parts[key]:
            return None
        return np.concatenate(parts[key]).astype(np.float32, copy=False)

    return Channels(y=joined("y"), r=joined("r"), s=joined("s"), a=joined("a"))


def _memmap_chunks(x_path, chunk_size):
    X_mm = np.load(x_path, mmap_mode="r")
    try:
        for start in range(0, X_mm.shape[0], chunk_size):
            # ascontiguousarray materialises just this slice; handing the memmap itself
            # to onnxruntime or the AD artifact would pull the whole matrix into memory.
            yield np.ascontiguousarray(X_mm[start : start + chunk_size])
    finally:
        del X_mm


def _featurized_chunks(featurizer, smiles_list, chunk_size):
    for start in range(0, len(smiles_list), chunk_size):
        yield featurizer.transform(smiles_list[start : start + chunk_size])


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
    return _score_chunks(
        _memmap_chunks(x_path, chunk_size), artifact, ad_artifact, want, logger
    )


def score_smiles_chunkwise(
    featurizer, artifact, ad_artifact, smiles_list, chunk_size, want, logger=None
):
    """Featurize and score in one pass, never holding the whole descriptor matrix.

    For a caller that already has the featurizer and the artifact in memory and is
    scoring one model, so there is nothing to gain from staging the matrix on disk. The
    previous behaviour — featurize everything, then score — needed roughly
    ``n_samples x n_features x 4`` bytes, which is about 8 GB for a million compounds
    against a 2048-dimensional descriptor.
    """
    return _score_chunks(
        _featurized_chunks(featurizer, smiles_list, chunk_size),
        artifact,
        ad_artifact,
        want,
        logger,
    )
