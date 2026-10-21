"""Classification assembler: preprocessor + heads + gating pooler, batched for imbalance.

The shared skeleton is :mod:`lazyqsar.assemblers._base`; this module adds what only
classification has: imbalance-aware batching with a prior correction, calibrated
probabilities, and the decision cutoff.
"""

import numpy as np

from lazyqsar.utils.ranking import (
    DECISION_CUTOFF_SOURCE,
    binarize,
    prepare_knots,
    rank_from_knots,
)

from ..portfolios.classification import Portfolio
from ..poolers.classification import InnerPooler
from ._base import _AssemblerBase, _BatchBase, _digest
from lazyqsar.utils.logging import logger
from lazyqsar.utils.splits import check_trainable


# One implementation, shared with the inference artifact so the two cannot drift.
from lazyqsar.artifacts.classifier import _correct_prior  # noqa: E402,F401


def _plan_batches(
    X, y, max_batch_size=100_000, max_imbalance_ratio=100, random_state=42
):
    """
    Returns a list of index arrays, one per batch.

    Balanced data (ratio <= max_imbalance_ratio):
      - n <= max_batch_size  ->  single batch (all indices)
      - n >  max_batch_size  ->  sequential slices of max_batch_size

    Imbalanced data (ratio > max_imbalance_ratio):
      - Shuffle negatives, partition into slices of max_imbalance_ratio * n_pos
      - Every batch contains ALL positives + one negative slice
      - Positives are class-1 samples; class-0 samples are the majority
    """
    n = len(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return [np.arange(n)]

    ratio = n_neg / n_pos

    if ratio <= max_imbalance_ratio:
        if n <= max_batch_size:
            return [np.arange(n)]
        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(n)
        return [
            shuffled[start : min(start + max_batch_size, n)]
            for start in range(0, n, max_batch_size)
        ]

    # Imbalanced path: all positives + equally-distributed negative partitions
    rng = np.random.default_rng(random_state)
    shuffled_neg = rng.permutation(neg_idx)
    n_batches = max(1, int(np.ceil(n_neg / (max_imbalance_ratio * n_pos))))
    neg_splits = np.array_split(shuffled_neg, n_batches)

    return [np.concatenate([pos_idx, split]) for split in neg_splits]


class _BatchLazyClassifier(_BatchBase):
    """
    Single-batch ensemble: preprocessor + heads + gating pooler.

    Trained on one balanced slice of the full dataset. Multiple instances
    are created by ``LazyClassifier`` when data is severely imbalanced or
    too large for a single batch.

    Parameters
    ----------
    portfolio : list of str
        Head names to train, e.g. ``["xgb", "rf"]``.
    calibrated : bool
        Whether to run OOF calibration on each head.
    max_rounds : int or None
        XGBoost round cap (None = use portfolio-selected value).
    """

    task = "classification"
    _pooler_cls = InnerPooler
    _oof_attr = "oof_probas_"

    def fit(self, X, y):
        """Fit preprocessor, all heads, and the gating pooler on (X, y)."""
        self.train_prior_ = float(np.mean(y == 1))
        self._fit_batch(X, y)

        def mean_of(attr):
            values = [
                getattr(h.model, attr)
                for h in self.heads
                if hasattr(getattr(h, "model", None), attr)
            ]
            return float(np.mean(values)) if values else 0.5

        self.decision_cutoff_raw_ = mean_of("decision_cutoff_raw_")
        self.decision_cutoff_proba_ = mean_of("decision_cutoff_proba_")
        self.decision_cutoff_rank_ = mean_of("decision_cutoff_rank_")
        _p = np.clip(self.decision_cutoff_proba_, 1e-7, 1.0 - 1e-7)
        self.decision_cutoff_logit_ = float(np.log(_p / (1.0 - _p)))

    def predict_proba(self, X):
        """Return calibrated probabilities, shape (n, 2)."""
        X_prep = self.prep.transform(X)
        R = np.column_stack([head.predict_proba(X_prep)[:, 1] for head in self.heads])
        return self.pooler.predict_proba(R, X_prep)

    def predict_score(self, X):
        """Return raw (pre-calibration) gated scores, shape (n, 2)."""
        X_prep = self.prep.transform(X)
        R = np.column_stack([head.predict_score(X_prep)[:, 1] for head in self.heads])
        W = self.pooler.get_weights(X_prep)
        score_1 = (W * R).sum(axis=1)
        return np.column_stack([1 - score_1, score_1])

    def _oof_percentile(self, X):
        """Return gated rank quantiles (0–1), shape (n, 2)."""
        X_prep = self.prep.transform(X)
        R = np.column_stack([head._oof_percentile(X_prep)[:, 1] for head in self.heads])
        W = self.pooler.get_weights(X_prep)
        rank_1 = (W * R).sum(axis=1)
        return np.column_stack([1 - rank_1, rank_1])

    def predict(self, X, cutoff=None):
        """Return binary labels using the OOF-learned decision cutoff."""
        threshold = self.decision_cutoff_raw_ if cutoff is None else cutoff
        return binarize(self.predict_score(X)[:, 1], threshold)


class LazyClassifier(_AssemblerBase):
    """
    Imbalance-aware ensemble classifier for pre-computed feature arrays.

    Wraps one or more ``_BatchLazyClassifier`` instances. For balanced data,
    a single batch is used. For severely imbalanced data (ratio > max_imbalance_ratio),
    multiple batches are created — each containing all positives and a disjoint
    subset of negatives — and their predictions are prior-corrected and averaged.

    Parameters
    ----------
    max_batch_size : int
        Maximum samples per batch for balanced data.
    calibrated : bool
        Whether to run OOF calibration on each head.
    max_rounds : int or None
        XGBoost round cap passed to each batch.
    max_imbalance_ratio : int
        Negative-to-positive ratio above which imbalanced batching is used.

    Attributes (after fit)
    ----------------------
    portfolio : list of str
        Head names selected by the Portfolio class.
    models : list of _BatchLazyClassifier
        One fitted batch per training slice.
    population_prior_ : float
        Fraction of positives in the full training set.
    oof_auc_ : float
        Out-of-fold AUC averaged across batches.
    train_auc_ : float
        AUC on the full training set (optimistic estimate).
    decision_cutoff_ : float
        OOF-learned decision threshold, averaged across batches.
    """

    task = "classification"

    def __init__(
        self,
        max_batch_size=100_000,
        calibrated=True,
        max_rounds=None,
        max_imbalance_ratio=100,
    ):
        self.max_batch_size = max_batch_size
        self.max_imbalance_ratio = max_imbalance_ratio
        self.calibrated = calibrated
        self.max_rounds = max_rounds

    def fit(self, X, y):
        """
        Fit the ensemble on (X, y).

        Selects a portfolio, plans batches, and fits one ``_BatchLazyClassifier``
        per batch in sequence.
        """
        logger.rule("LazyClassifier — fit")
        check_trainable(y, where="LazyClassifier.fit")
        self.population_prior_ = float(np.mean(y == 1))

        p = Portfolio()
        p.fit(X, y)
        self.portfolio = p.get()
        logger.dataset_table(X.shape, y=y, portfolio=self.portfolio)

        batch_indices = _plan_batches(
            X, y, self.max_batch_size, self.max_imbalance_ratio
        )

        n_pos_total = int((y == 1).sum())
        n_neg_total = int((y == 0).sum())
        ratio = n_neg_total / max(n_pos_total, 1)
        strategy = "imbalanced" if ratio > self.max_imbalance_ratio else "sequential"
        batch_details = [
            {
                "n": len(idx),
                "n_pos": int((y[idx] == 1).sum()),
                "n_neg": int((y[idx] == 0).sum()),
            }
            for idx in batch_indices
        ]
        logger.batch_table(batch_details, strategy=strategy)

        n_batches = len(batch_indices)
        self.models = []
        for batch_idx, indices in enumerate(batch_indices):
            batch_X = X[indices]
            batch_y = y[indices]
            logger.info(
                f"Batch {batch_idx + 1}/{n_batches} — "
                f"n={len(batch_X):,}  portfolio={self.portfolio}"
            )
            batch_classifier = _BatchLazyClassifier(
                portfolio=self.portfolio,
                calibrated=self.calibrated,
                max_rounds=self.max_rounds,
            )
            batch_classifier.fit(batch_X, batch_y)
            self.models.append(batch_classifier)

        # Kept, not just used: `oof_channels` replays the same partition to build the
        # pooled out-of-fold reference. Re-deriving it there would couple that reference
        # to `_plan_batches`' private seed.
        self.batch_indices_ = batch_indices
        self.batch_priors_ = [m.train_prior_ for m in self.models]
        self.decision_cutoff_raw_ = float(
            np.mean([m.decision_cutoff_raw_ for m in self.models])
        )
        self.decision_cutoff_proba_ = float(
            np.mean([m.decision_cutoff_proba_ for m in self.models])
        )
        self.decision_cutoff_rank_ = float(
            np.mean([m.decision_cutoff_rank_ for m in self.models])
        )
        _p = np.clip(self.decision_cutoff_proba_, 1e-7, 1.0 - 1e-7)
        self.decision_cutoff_logit_ = float(np.log(_p / (1.0 - _p)))
        _prior = self.population_prior_
        self.decision_cutoff_lift_ = (
            float(self.decision_cutoff_proba_ / _prior)
            if _prior and _prior > 0
            else None
        )
        self._build_oof_percentile(X)
        self.oof_auc_ = self._compute_oof_auc(X, y, batch_indices)
        self.train_auc_ = self._compute_train_auc(X, y)

        def _auc(value):
            return "unknown" if value is None else f"{value:.4f}"

        logger.success(
            f"LazyClassifier fitted — "
            f"{n_batches} batch(es)  portfolio={self.portfolio}  "
            f"OOF AUC={_auc(self.oof_auc_)}  train AUC={_auc(self.train_auc_)}"
        )

    def _compute_train_auc(self, X, y) -> float:
        """Training-set AUC, or ``None`` when it could not be computed.

        It used to return 0.5 on failure, which is indistinguishable from a genuine
        coin-flip AUC and is *rewarded* downstream: `LazyClassifierQSAR` derives
        ``quality = 2*oof - train``, so a failed computation on a strong descriptor gives
        it a quality above 1 and the largest weight in the ensemble. `None` propagates the
        ignorance instead, and the failure is logged rather than swallowed.
        """
        from sklearn.metrics import roc_auc_score

        try:
            proba = self.predict_proba(X)
            # The QSAR layer asks for the training-set probabilities again; keep them.
            self._train_proba_ = (_digest(X), X.shape, proba)
            return float(roc_auc_score(y, proba[:, 1]))
        except Exception as exc:  # noqa: BLE001 - reported, then degraded deliberately
            logger.warning(
                f"Training AUC could not be computed ({type(exc).__name__}: {exc}); "
                "this descriptor's quality weight falls back to its out-of-fold AUC."
            )
            return None

    def _build_oof_percentile(self, X):
        """Learn the out-of-fold probability distribution ``_oof_percentile`` reports against.

        Sets ``oof_percentile_knots_``, which is persisted, and its prepared form. Both stay
        ``None`` when out-of-fold data is unavailable, and ``_oof_percentile`` then keeps its
        pre-v3.5.0 behaviour.
        """
        pooled = self._oof_accumulate(X)
        self._set_oof_reference(None if pooled is None else pooled[0])

    def _oof_accumulate(self, X):
        """Pooled out-of-fold ``(proba, score)`` per training row, or None.

        The pooled reference that ``rank`` is measured against has to be built from
        predictions the model did not train on, or the ECDF is calibrated against a
        distribution nothing at inference time produces. The pieces exist -- every head
        keeps ``oof_probas_`` and ``oof_raw_`` from its calibration fold -- but they live
        one batch and one head down, and until now were read once for an AUC and thrown
        away.

        Each value is combined exactly as its deployed counterpart is: heads are pooled
        with the batch's gating weights, as :meth:`_BatchLazyClassifier.predict_proba`
        does, and the prior correction is applied per batch *before* averaging, as
        :meth:`predict_proba` does. Applying it after averaging is a different function.

        Returns ``None`` when any head lacks out-of-fold data. A partial reference is
        worse than none: a mis-calibrated ECDF is still monotone and still in [0, 1], so
        nothing downstream could detect it.

        Notes
        -----
        Under imbalance batching a positive appears in every batch and a negative in
        exactly one, so a negative's value here comes from a single sub-model where
        deployment averages all of them. Averaging over the batches that contain each row
        is the only defensible reading; it leaves the reference's tails mildly optimistic.
        """
        indices = getattr(self, "batch_indices_", None)
        if not indices or not getattr(self, "models", None):
            return None

        n = X.shape[0]
        acc_y = np.zeros(n, dtype=np.float64)
        acc_s = np.zeros(n, dtype=np.float64)
        cnt = np.zeros(n, dtype=np.float64)

        for batch, idx in zip(self.models, indices):
            heads = [getattr(h, "model", None) for h in batch.heads]
            if not all(
                m is not None and hasattr(m, "oof_probas_") and hasattr(m, "oof_raw_")
                for m in heads
            ):
                return None
            try:
                S_cal = np.column_stack([m.oof_probas_ for m in heads])
                S_raw = np.column_stack([m.oof_raw_ for m in heads])
                W = batch.training_weights(X[idx])
            except Exception as exc:  # noqa: BLE001 - deliberately broad; degrade, don't raise
                logger.debug(
                    f"pooled rank reference unavailable: {type(exc).__name__}: {exc}"
                )
                return None
            if S_cal.shape[0] != len(idx):
                return None

            acc_y[idx] += _correct_prior(
                (W * S_cal).sum(axis=1), batch.train_prior_, self.population_prior_
            )
            acc_s[idx] += (W * S_raw).sum(axis=1)
            cnt[idx] += 1.0

        if (cnt == 0).any():
            return None
        return acc_y / cnt, acc_s / cnt

    def oof_channels(self, X):
        """Out-of-fold ``(proba, rank, score)`` per training row, or None.

        What :class:`~lazyqsar.qsar.LazyClassifierQSAR` stacks across descriptors to build
        the ensemble-level pooled reference. ``rank`` is this descriptor's own reference
        evaluated at ``proba``, so it is uniform by construction -- which is what makes it
        a usable self-check on the reference rather than a second opinion about it.
        """
        prepared = getattr(self, "_oof_percentile_prepared_", None)
        pooled = self._oof_accumulate(X)
        if pooled is None or prepared is None:
            return None
        proba, score = pooled
        return proba, rank_from_knots(proba, prepared=prepared), score

    def _compute_oof_auc(self, X, y, batch_indices) -> float:
        from sklearn.metrics import roc_auc_score

        try:
            batch_aucs = []
            for batch_clf, indices in zip(self.models, batch_indices):
                heads = batch_clf.heads
                if not all(
                    hasattr(getattr(h, "model", None), "oof_probas_") for h in heads
                ):
                    logger.warning(
                        "A head in this batch kept no out-of-fold probabilities, so the "
                        "out-of-fold AUC is unknown for this descriptor."
                    )
                    return None
                S = np.column_stack([h.model.oof_probas_ for h in heads])
                W_oof = batch_clf.training_weights(X[indices])
                pooled = (W_oof * S).sum(axis=1)
                batch_aucs.append(roc_auc_score(y[indices], pooled))
            return float(np.mean(batch_aucs))
        except Exception as exc:  # noqa: BLE001 - reported, then degraded deliberately
            # Not 0.5: that sits below the 0.55 pruning floor in `LazyClassifierQSAR`, so
            # a failed computation silently *deleted* the descriptor rather than
            # admitting it did not know.
            logger.warning(
                f"Out-of-fold AUC could not be computed ({type(exc).__name__}: {exc}); "
                "this descriptor is kept, with an unknown out-of-fold AUC."
            )
            return None

    def predict_proba(self, X):
        """Return prior-corrected calibrated probabilities, shape (n, 2)."""
        cached = getattr(self, "_train_proba_", None)
        if cached is not None and cached[1] == X.shape and cached[0] == _digest(X):
            # Exactly the training rows: the same numbers, without another full pass.
            return cached[2].copy()
        logger.debug(f"predict_proba: X={X.shape}  batches={len(self.models)}")
        R = np.array(
            [
                _correct_prior(m.predict_proba(X)[:, 1], tp, self.population_prior_)
                for m, tp in zip(self.models, self.batch_priors_)
            ]
        )
        proba = R.mean(axis=0)
        return np.array([1 - proba, proba]).T

    def predict_lift(self, X) -> np.ndarray:
        """Return lift over population prior, shape (n_samples, 2)."""
        proba = self.predict_proba(X)
        return np.column_stack(
            [
                proba[:, 0] / (1.0 - self.population_prior_),
                proba[:, 1] / self.population_prior_,
            ]
        )

    def predict_logit(self, X):
        """Return log-odds of calibrated probabilities, shape (n, 2)."""
        p = np.clip(self.predict_proba(X)[:, 1], 1e-7, 1.0 - 1e-7)
        logit_1 = np.log(p / (1.0 - p))
        return np.column_stack([-logit_1, logit_1])

    def predict_score(self, X):
        """Return batch-averaged raw (pre-calibration) scores, shape (n, 2)."""
        R = np.array([model.predict_score(X)[:, 1] for model in self.models])
        proba = R.mean(axis=0)
        return np.array([1 - proba, proba]).T

    def _oof_percentile(self, X):
        """Return the training-set percentile of the pooled probability, shape (n, 2).

        Not an average of the batches' percentiles: averaging percentiles does not give
        the percentile of the average, so that version could order two molecules one way
        while ``predict_proba`` ordered them the other. Reading one pooled reference makes
        ``rank`` a monotone view of ``proba``, which is what it was always taken to be --
        and it removes the ECDF amplification that made ``rank`` the output least
        reproducible across the ONNX export boundary.

        Falls back to the batch average for models fitted before the reference existed.
        """
        prepared = getattr(self, "_oof_percentile_prepared_", None)
        if prepared is None:
            knots = getattr(self, "oof_percentile_knots_", None)
            if knots is not None and len(knots):
                prepared = self._oof_percentile_prepared_ = prepare_knots(knots)
        if prepared is not None:
            rank_1 = rank_from_knots(self.predict_proba(X)[:, 1], prepared=prepared)
            return np.column_stack([1 - rank_1, rank_1])
        R = np.array([model._oof_percentile(X)[:, 1] for model in self.models])
        rank_1 = R.mean(axis=0)
        return np.column_stack([1 - rank_1, rank_1])

    def predict(self, X, cutoff=None):
        """Return binary labels using this model's decision cutoff.

        Two paths, branching on what the cutoff *is* rather than on any setting:

        - A cutoff placed on the reference rank scale exists in probability units only, so
          it is compared against ``predict_proba``. The model then calls a known fraction of
          drug-like chemical space active.
        - Without a reference there is no such cutoff, and the out-of-fold
          balanced-accuracy threshold is compared against ``predict_score`` exactly as
          before, so a model fitted without one labels bit-identically.
        """
        if cutoff is None and getattr(self, "decision_cutoff_source_", None) == (
            DECISION_CUTOFF_SOURCE
        ):
            return binarize(self.predict_proba(X)[:, 1], self.decision_cutoff_proba_)
        threshold = self.decision_cutoff_raw_ if cutoff is None else cutoff
        return binarize(self.predict_score(X)[:, 1], threshold)

    def _metadata(self) -> dict:
        metadata = {
            "portfolio": self.portfolio,
            "max_batch_size": self.max_batch_size,
            "max_imbalance_ratio": self.max_imbalance_ratio,
            "num_batches": len(self.models),
            "population_prior": self.population_prior_,
            "batch_priors": self.batch_priors_,
            "decision_cutoff_raw": self.decision_cutoff_raw_,
            "decision_cutoff_proba": self.decision_cutoff_proba_,
            # Named for what it is: a percentile against this model's own out-of-fold
            # scores, not a position against the reference library. The task-level
            # `decision_cutoff_rank` is the latter, and the two must not be confused.
            "decision_cutoff_oof_percentile": self.decision_cutoff_rank_,
            "decision_cutoff_logit": self.decision_cutoff_logit_,
            "decision_cutoff_lift": self.decision_cutoff_lift_,
            # Names which of the two cutoff conventions this checkpoint carries, so the
            # artifact's `predict` can branch on data rather than guessing from the value.
            "decision_cutoff_source": getattr(self, "decision_cutoff_source_", None),
        }
        reference_rank = getattr(self, "decision_cutoff_reference_rank_", None)
        if reference_rank is not None:
            # Only when the cutoff was placed against a reference library, and under the
            # same key the task level uses for that position.
            metadata["decision_cutoff_rank"] = float(reference_rank)
        return metadata

    def _extend_metadata(self, metadata: dict) -> None:
        ref = getattr(self, "reference_rank_knots_", None)
        if ref is not None and len(ref):
            # `pooled_ranker` means the reference library wherever it appears, so the
            # agnostic path writes the same block the task level does and is read by the
            # same code. The out-of-fold percentile lives under `oof_percentile`.
            metadata["pooled_ranker"] = {
                "knots": np.asarray(ref, dtype=np.float64).tolist(),
                "n_train": int(len(ref)),
                "source": "reference_library",
            }
            anchors = getattr(self, "reference_rank_anchors_", None)
            if anchors:
                metadata["pooled_ranker"].update(anchors)

    def load(self, directory):
        raise NotImplementedError("Loading not implemented for LazyClassifier yet.")
