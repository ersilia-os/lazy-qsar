import json
import os
import time as _time
import numpy as np

from ..portfolios.classification import Portfolio
from ..preprocessors.classification.prep import Preprocessor
from ..heads.classification.lr import Head as LRHead
from ..heads.classification.xgb import Head as XGBHead
from ..heads.classification.rf import Head as RFHead
from ..poolers.classification import InnerPooler
from lazyqsar.utils.logging import logger


def _correct_prior(p1, train_prior, population_prior):
    """Adjust P(y=1) trained at train_prior to reflect population_prior."""
    if abs(train_prior - population_prior) < 1e-9:
        return p1
    if train_prior <= 0.0 or train_prior >= 1.0:
        return p1
    ratio = (population_prior / train_prior) / (
        (1.0 - population_prior) / (1.0 - train_prior)
    )
    odds = p1 / np.clip(1.0 - p1, 1e-15, None)
    corrected_odds = ratio * odds
    return corrected_odds / (1.0 + corrected_odds)


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
        return [
            np.arange(start, min(start + max_batch_size, n))
            for start in range(0, n, max_batch_size)
        ]

    # Imbalanced path: all positives + equally-distributed negative partitions
    rng = np.random.default_rng(random_state)
    shuffled_neg = rng.permutation(neg_idx)
    n_batches = max(1, int(np.ceil(n_neg / (max_imbalance_ratio * n_pos))))
    neg_splits = np.array_split(shuffled_neg, n_batches)

    return [np.concatenate([pos_idx, split]) for split in neg_splits]


class _BatchLazyClassifier(object):
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

    def __init__(
        self, portfolio: list, calibrated: bool = True, max_rounds: int | None = None
    ):
        self.prep = Preprocessor()
        self.heads = []
        for head_name in portfolio:
            if head_name == "lr":
                self.heads += [LRHead(calibrated=calibrated)]
            elif head_name == "xgb":
                self.heads += [XGBHead(calibrated=calibrated, max_rounds=max_rounds)]
            elif head_name == "rf":
                self.heads += [RFHead(calibrated=calibrated)]
            else:
                raise ValueError(f"Unknown head {head_name}.")
        self.portfolio = portfolio
        self.pooler = InnerPooler(portfolio=portfolio)

    def fit(self, X, y):
        """Fit preprocessor, all heads, and the gating pooler on (X, y)."""
        self.train_prior_ = float(np.mean(y == 1))
        _t_prep = _time.perf_counter()
        self.prep.fit(X, y)
        X = self.prep.transform(X)
        t_prep = _time.perf_counter() - _t_prep

        for i, head in enumerate(self.heads):
            logger.info(f"Fitting head {i + 1}/{len(self.heads)}: {self.portfolio[i]}")
            head.fit(X, y)
        if all(hasattr(getattr(h, "model", None), "oof_probas_") for h in self.heads):
            S = np.column_stack([h.model.oof_probas_ for h in self.heads])
        else:
            S = None
        _t_pooler = _time.perf_counter()
        self.pooler.fit(S, y, X_prep=X)
        t_pooler = _time.perf_counter() - _t_pooler
        cutoffs = [
            h.model.decision_cutoff_
            for h in self.heads
            if hasattr(getattr(h, "model", None), "decision_cutoff_")
        ]
        self.decision_cutoff_ = float(np.mean(cutoffs)) if cutoffs else 0.5

        # Build and display per-step timing table
        steps = [("Preprocessing", t_prep, False)]
        for head_name, head in zip(self.portfolio, self.heads):
            t = getattr(getattr(head, "model", None), "timing_", {})
            if head_name == "xgb":
                if "portfolio_select" in t:
                    steps.append(
                        (
                            "XGB \u2014 portfolio select (stage 1+2)",
                            t["portfolio_select"],
                            False,
                        )
                    )
                steps.append(
                    ("XGB \u2014 phase-2 refit", t.get("phase2_refit", 0.0), False)
                )
                if "calibration_total" in t:
                    folds = t.get("calibration_folds", [])
                    steps.append(
                        (
                            f"XGB \u2014 calibration ({len(folds)} folds)",
                            t["calibration_total"],
                            False,
                        )
                    )
                    for fi, ft in enumerate(folds):
                        steps.append((f"fold {fi + 1}/{len(folds)}", ft, True))
            elif head_name == "lr":
                steps.append(
                    ("LR \u2014 hyperparam search", t.get("hparam_search", 0.0), False)
                )
                if "calibration_total" in t:
                    folds = t.get("calibration_folds", [])
                    steps.append(
                        (
                            f"LR \u2014 calibration ({len(folds)} folds)",
                            t["calibration_total"],
                            False,
                        )
                    )
                    for fi, ft in enumerate(folds):
                        steps.append((f"fold {fi + 1}/{len(folds)}", ft, True))
            elif head_name == "rf":
                steps.append(("RF \u2014 fit", t.get("fit", 0.0), False))
                if "calibration_total" in t:
                    folds = t.get("calibration_folds", [])
                    steps.append(
                        (
                            f"RF \u2014 calibration ({len(folds)} folds)",
                            t["calibration_total"],
                            False,
                        )
                    )
                    for fi, ft in enumerate(folds):
                        steps.append((f"fold {fi + 1}/{len(folds)}", ft, True))
        steps.append(("Pooler \u2014 gating network", t_pooler, False))
        logger.timing_table(steps)

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

    def predict_rank(self, X):
        """Return gated rank quantiles (0–1), shape (n, 2)."""
        X_prep = self.prep.transform(X)
        R = np.column_stack([head.predict_rank(X_prep)[:, 1] for head in self.heads])
        W = self.pooler.get_weights(X_prep)
        rank_1 = (W * R).sum(axis=1)
        return np.column_stack([1 - rank_1, rank_1])

    def predict(self, X, cutoff=None):
        """Return binary labels using the OOF-learned decision cutoff."""
        threshold = self.decision_cutoff_ if cutoff is None else cutoff
        return (self.predict_score(X)[:, 1] >= threshold).astype(int)

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


class LazyClassifier(object):
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

        self.batch_priors_ = [m.train_prior_ for m in self.models]
        self.decision_cutoff_ = float(
            np.mean([m.decision_cutoff_ for m in self.models])
        )
        self.oof_auc_ = self._compute_oof_auc(X, y, batch_indices)
        self.train_auc_ = self._compute_train_auc(X, y)
        logger.success(
            f"LazyClassifier fitted — "
            f"{n_batches} batch(es)  portfolio={self.portfolio}  "
            f"OOF AUC={self.oof_auc_:.4f}  train AUC={self.train_auc_:.4f}"
        )

    def _compute_train_auc(self, X, y) -> float:
        from sklearn.metrics import roc_auc_score

        try:
            train_proba = self.predict_proba(X)[:, 1]
            return float(roc_auc_score(y, train_proba))
        except Exception:
            return 0.5

    def _compute_oof_auc(self, X, y, batch_indices) -> float:
        from sklearn.metrics import roc_auc_score

        try:
            batch_aucs = []
            for batch_clf, indices in zip(self.models, batch_indices):
                heads = batch_clf.heads
                if not all(
                    hasattr(getattr(h, "model", None), "oof_probas_") for h in heads
                ):
                    return 0.5
                S = np.column_stack([h.model.oof_probas_ for h in heads])
                X_prep = batch_clf.prep.transform(X[indices])
                W_oof = batch_clf.pooler.get_weights(X_prep)
                pooled = (W_oof * S).sum(axis=1)
                batch_aucs.append(roc_auc_score(y[indices], pooled))
            return float(np.mean(batch_aucs))
        except Exception:
            return 0.5

    def predict_proba(self, X):
        """Return prior-corrected calibrated probabilities, shape (n, 2)."""
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

    def predict_rank(self, X):
        """Return batch-averaged rank quantiles (0–1), shape (n, 2)."""
        R = np.array([model.predict_rank(X)[:, 1] for model in self.models])
        rank_1 = R.mean(axis=0)
        return np.column_stack([1 - rank_1, rank_1])

    def predict(self, X, cutoff=None):
        """Return binary labels using the OOF-learned decision cutoff."""
        threshold = self.decision_cutoff_ if cutoff is None else cutoff
        return (self.predict_score(X)[:, 1] >= threshold).astype(int)

    def save(self, directory):
        """Save all batch models and metadata.json to *directory*."""
        logger.info(f"Saving LazyClassifier to {directory!r}")
        for batch_num, model in enumerate(self.models):
            model.save(directory, batch_num)
        metadata = {
            "portfolio": self.portfolio,
            "max_batch_size": self.max_batch_size,
            "max_imbalance_ratio": self.max_imbalance_ratio,
            "num_batches": len(self.models),
            "population_prior": self.population_prior_,
            "batch_priors": self.batch_priors_,
            "decision_cutoff": self.decision_cutoff_,
        }
        with open(f"{directory}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        logger.success(f"Saved {len(self.models)} batch(es) to {directory!r}")
        logger.dir_tree(directory)

    def load(self, directory):
        raise NotImplementedError("Loading not implemented for LazyClassifier yet.")
