"""The assembler skeleton both tasks share.

An assembler fits, per batch of training rows, a preprocessor, the portfolio's heads and
a gating pooler over the heads' out-of-fold outputs; across batches it averages. What
differs by task -- how rows are batched, what is pooled (calibrated probabilities with a
prior correction, or standardised values), the decision cutoff -- stays in
:mod:`~lazyqsar.assemblers.classifier` and :mod:`~lazyqsar.assemblers.regressor`.
"""

from __future__ import annotations

import json
import os
import time as _time

import numpy as np

from lazyqsar.heads import HEAD_LABELS, Head, head_kwargs
from lazyqsar.preprocessors import Preprocessor
from lazyqsar.utils.logging import logger
from lazyqsar.utils.ranking import prepare_knots, subsample_knots


def _digest(X) -> bytes:
    """A fingerprint of a feature matrix: its shape, dtype and bytes."""
    import hashlib

    if hasattr(X, "toarray"):
        X = X.toarray()
    arr = np.ascontiguousarray(X)
    h = hashlib.blake2b(digest_size=16)
    h.update(str((arr.shape, arr.dtype.str)).encode())
    h.update(arr.view(np.uint8).ravel())
    return h.digest()


def timing_steps(t_prep, portfolio, heads, t_pooler) -> list:
    """Rows of the per-batch timing table: preprocessing, each head's stages, the gate."""
    steps = [("Preprocessing", t_prep, False)]
    for head_name, head in zip(portfolio, heads):
        label = HEAD_LABELS.get(head_name, head_name)
        t = getattr(getattr(head, "model", None), "timing_", {})
        if "portfolio_select" in t:
            steps.append((f"{label} — portfolio select", t["portfolio_select"], False))
        if "hparam_search" in t:
            steps.append((f"{label} — hyperparam search", t["hparam_search"], False))
        if "phase2_refit" in t:
            steps.append((f"{label} — phase-2 refit", t["phase2_refit"], False))
        if "fit" in t:
            steps.append((f"{label} — fit", t["fit"], False))
        for key, title in (("calibration", "calibration"), ("oof", "OOF")):
            if f"{key}_total" in t:
                folds = t.get(f"{key}_folds", [])
                steps.append(
                    (
                        f"{label} — {title} ({len(folds)} folds)",
                        t[f"{key}_total"],
                        False,
                    )
                )
                for fi, ft in enumerate(folds):
                    steps.append((f"fold {fi + 1}/{len(folds)}", ft, True))
    steps.append(("Pooler — gating network", t_pooler, False))
    return steps


class _BatchBase:
    """One batch: preprocessor + heads + gating pooler.

    A subclass names its ``task``, its pooler class (``_pooler_cls``) and the head
    attribute holding the out-of-fold output the gate is fitted on (``_oof_attr``).
    """

    task = ""
    _pooler_cls = None
    _oof_attr = ""

    def __init__(
        self, portfolio: list, calibrated: bool = True, max_rounds: int | None = None
    ):
        self.prep = Preprocessor(self.task)
        self.heads = [
            Head(
                self.task, name, **head_kwargs(self.task, name, calibrated, max_rounds)
            )
            for name in portfolio
        ]
        self.portfolio = portfolio
        self.pooler = self._pooler_cls(portfolio=portfolio)

    def _fit_batch(self, X, y) -> None:
        """Fit the preprocessor, every head, then the gate on the heads' OOF outputs."""
        raw_digest = _digest(X)
        t0 = _time.perf_counter()
        self.prep.fit(X, y)
        X = self.prep.transform(X)
        t_prep = _time.perf_counter() - t0

        for i, head in enumerate(self.heads):
            logger.info(f"Fitting head {i + 1}/{len(self.heads)}: {self.portfolio[i]}")
            head.fit(X, y)
        if all(hasattr(getattr(h, "model", None), self._oof_attr) for h in self.heads):
            oof = np.column_stack(
                [getattr(h.model, self._oof_attr) for h in self.heads]
            )
        else:
            oof = None
        t0 = _time.perf_counter()
        self.pooler.fit(oof, y, X_prep=X)
        t_pooler = _time.perf_counter() - t0
        # The gate's weights on the training rows, which the out-of-fold statistics ask
        # for several times; kept with a digest of the rows they belong to.
        self._train_weights_ = self.pooler.get_weights(X)
        self._train_digest_ = raw_digest
        logger.timing_table(timing_steps(t_prep, self.portfolio, self.heads, t_pooler))

    def training_weights(self, X) -> np.ndarray:
        """Gate weights for rows *X*: the stored ones if *X* is exactly the training rows.

        Bit-identical either way -- the stored weights came from the same preprocessor
        and gate on the same rows -- but the training rows are not pushed through the
        preprocessor and gate again for every out-of-fold statistic.
        """
        cached = getattr(self, "_train_weights_", None)
        if cached is not None and len(cached) == X.shape[0]:
            if _digest(X) == getattr(self, "_train_digest_", None):
                return cached
        return self.pooler.get_weights(self.prep.transform(X))

    def save(self, directory, batch_num):
        """Save this batch to ``{directory}/batch_{batch_num}/``."""
        if batch_num is not None:
            directory = f"{directory}/batch_{batch_num}"
        os.makedirs(directory, exist_ok=True)
        self.pooler.save(directory)
        self.prep.save(directory)
        for head in self.heads:
            head.save(directory)
        logger.debug(f"Batch saved to {directory}")
        return directory


class _AssemblerBase:
    """Batches, the pooled out-of-fold reference, and ``save``.

    A subclass provides ``_metadata()``: the task-specific entries of ``metadata.json``.
    """

    task = ""

    def _set_oof_reference(self, pooled) -> None:
        """ECDF knots over the pooled out-of-fold outputs (``None`` when there are none)."""
        if pooled is None:
            self.oof_percentile_knots_ = None
            self._oof_percentile_prepared_ = None
            return
        self.oof_percentile_knots_ = subsample_knots(np.sort(pooled))
        self._oof_percentile_prepared_ = prepare_knots(self.oof_percentile_knots_)

    def _metadata(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def save(self, directory):
        """Save all batch models and ``metadata.json`` to *directory*."""
        logger.info(f"Saving {type(self).__name__} to {directory!r}")
        for batch_num, model in enumerate(self.models):
            model.save(directory, batch_num)
        metadata = {"task": self.task, **self._metadata()}
        knots = getattr(self, "oof_percentile_knots_", None)
        if knots is not None and len(knots):
            # The out-of-fold percentile reference. Not `pooled_ranker`: that name means
            # the reference library wherever it appears.
            metadata["oof_percentile"] = {
                "knots": np.asarray(knots, dtype=np.float64).tolist(),
                "n_train": int(len(knots)),
            }
        self._extend_metadata(metadata)
        with open(f"{directory}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        logger.success(f"Saved {len(self.models)} batch(es) to {directory!r}")
        logger.dir_tree(directory)

    def _extend_metadata(self, metadata: dict) -> None:
        """Entries that must follow the out-of-fold block (none by default)."""
