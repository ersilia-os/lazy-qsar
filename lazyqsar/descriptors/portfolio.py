import numpy as np

from ..utils.logging import logger

_REFERENCE_DESCRIPTOR = {"fast": "morgan", "slow": "chemeleon"}
_MAX_DESCRIPTORS = 3  # max descriptors kept after greedy selection
_IMBALANCE_THRESHOLD = 100  # matches LazyClassifier batching trigger


class DescriptorPortfolio:
    """Select applicable descriptors for a given SMILES dataset.

    Parameters
    ----------
    mode : str
        "fast" (morgan) or "slow" (chemeleon, clamp, morgan, rdkit, cddd).

    Example
    -------
    portfolio = DescriptorPortfolio("slow")
    for name, desc, X, proxy_auc in portfolio.select(smiles_list, y=y):
        ...
    """

    def __init__(self, mode: str):
        from ..registry import DESCRIPTORS_MODE

        assert mode in ("fast", "slow"), (
            f"Mode '{mode}' not recognized. Choose from 'fast' or 'slow'."
        )
        self.mode = mode
        self._all_names = DESCRIPTORS_MODE[mode]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        smiles_list: list,
        y=None,
        precomputed: dict = None,
        task: str = "classification",
    ) -> list:
        """Return applicable descriptors, optionally screened by proxy CV AUC.

        Step 1 — unsupervised gate: ``is_applicable(smiles_list)``
        Step 2 — supervised gate (only when *y* is provided and >1 descriptor
                  passes step 1): reference-anchored greedy forward selection
                  using OOF predictions from a 3-fold Random Forest CV.

        Parameters
        ----------
        smiles_list : list
            Compounds to profile.
        y : array-like, optional
            Labels. Screening is skipped without them.
        precomputed : dict, optional
            ``{descriptor_name: feature_matrix}`` aligned with *smiles_list*. Supplied by
            callers that have already featurized — the multi-task CLI fit computes each
            descriptor once over the union of every task, and would otherwise pay for it
            again per task, turning one featurization pass into one per task.

        task : str
            ``"classification"`` (default) or ``"regression"``. Regression screens with
            a random-forest regressor and scores by Spearman ρ instead of AUC, with
            folds stratified on quantile bins of *y*.

        Returns
        -------
        list of ``(name, descriptor_instance, X_full, proxy_score)`` tuples where
        ``X_full`` is the raw feature matrix computed during screening (or None),
        and ``proxy_score`` is the solo OOF AUC -- or Spearman ρ for regression -- or
        None when screening was skipped.
        """
        precomputed = precomputed or {}
        from ..registry import get_descriptor_type

        logger.info(
            f"Descriptor portfolio — mode='{self.mode}', "
            f"{len(smiles_list):,} SMILES, "
            f"candidates: {self._all_names}"
        )

        # --- Step 1: is_applicable ---
        applicable = []
        for name in self._all_names:
            desc = get_descriptor_type(name)()
            if desc.is_applicable(smiles_list):
                logger.info(f"  [{name}] applicable ✓")
                applicable.append((name, desc))
            else:
                logger.warning(
                    f"  [{name}] NOT applicable (too many out-of-domain SMILES) — skipping"
                )

        logger.info(
            f"  {len(applicable)}/{len(self._all_names)} descriptors passed applicability gate"
        )

        if not applicable:
            logger.warning("No descriptors are applicable. Using all as fallback.")
            applicable = [
                (name, get_descriptor_type(name)()) for name in self._all_names
            ]

        # --- Step 2: proxy CV screening (supervised, optional) ---
        if y is None or len(applicable) <= 1:
            reason = "y not provided" if y is None else "single descriptor"
            logger.info(f"  Skipping proxy CV screening ({reason})")
            screen_rows = [
                {
                    "name": name,
                    "n_features": None,
                    "proxy_auc": None,
                    "status": "skipped",
                }
                for name, _ in applicable
            ]
            logger.proxy_screen_table(
                screen_rows, metric="ρ" if task == "regression" else "AUC"
            )
            # Pass any precomputed matrix straight back so the caller's cache is
            # still primed when screening is skipped.
            return [
                (name, desc, precomputed.get(name), None) for name, desc in applicable
            ]

        if task == "regression":
            return self._proxy_screen_regression(
                smiles_list, np.asarray(y, dtype=float), applicable, precomputed
            )
        return self._proxy_screen(
            smiles_list, np.asarray(y, dtype=int), applicable, precomputed
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _proxy_screen(
        self,
        smiles_list: list,
        y: np.ndarray,
        applicable: list,
        precomputed: dict = None,
    ) -> list:
        from ..preprocessors import Preprocessor
        from joblib import parallel_backend
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        from sklearn.metrics import roc_auc_score
        from sklearn.ensemble import RandomForestClassifier

        # --- Balanced batch for imbalanced datasets ---
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        ratio = n_neg / max(n_pos, 1)

        logger.info(
            f"  Dataset: {n_pos:,} positives, {n_neg:,} negatives, ratio {ratio:.1f}:1"
        )

        if ratio > _IMBALANCE_THRESHOLD:
            rng = np.random.default_rng(42)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            n_neg_batch = min(len(neg_idx), _IMBALANCE_THRESHOLD * n_pos)
            neg_sample = rng.choice(neg_idx, size=n_neg_batch, replace=False)
            batch_idx = np.concatenate([pos_idx, neg_sample])
            logger.info(
                f"  Imbalance ratio > {_IMBALANCE_THRESHOLD} — "
                f"using balanced batch of {len(batch_idx):,} samples "
                f"({n_pos:,} pos + {n_neg_batch:,} neg)"
            )
        else:
            batch_idx = np.arange(len(y))
            logger.info(
                f"  Using full dataset ({len(batch_idx):,} samples) for proxy CV"
            )

        y_batch = y[batch_idx]

        proxy_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            n_jobs=1,
            random_state=42,
            class_weight="balanced",
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # --- Step 1: OOF predictions for every applicable descriptor ---
        logger.rule("Descriptor screening — proxy CV")
        oof_preds = {}  # name → np.ndarray (n_batch,)
        solo_aucs = {}  # name → float
        feature_map = {}  # name → X_full (raw, full dataset)

        for name, desc in applicable:
            X_full = (precomputed or {}).get(name)
            if X_full is None:
                logger.info(
                    f"  [{name}] Computing features for {len(smiles_list):,} SMILES..."
                )
                X_full = desc.transform(smiles_list)
            else:
                logger.debug(f"  [{name}] Using precomputed features")
            feature_map[name] = X_full
            logger.info(
                f"  [{name}] Feature matrix: {X_full.shape[0]:,} × {X_full.shape[1]:,}"
            )

            X_batch = X_full[batch_idx]
            prep = Preprocessor("classification")
            prep.fit(X_batch, y_batch)
            X_prep = prep.transform(X_batch)
            logger.info(
                f"  [{name}] Preprocessed features: {X_prep.shape[1]:,} (after reduction)"
            )

            # The folds in threads: each forest stays single-threaded and seeded, so the
            # predictions are those of the sequential run.
            with parallel_backend("threading", n_jobs=3):
                proba = cross_val_predict(
                    proxy_model,
                    X_prep,
                    y_batch,
                    cv=cv,
                    method="predict_proba",
                    n_jobs=3,
                )[:, 1]
            oof_preds[name] = proba
            auc = float(roc_auc_score(y_batch, proba))
            solo_aucs[name] = auc
            logger.info(f"  [{name}] solo proxy AUC = {auc:.4f}")

        # --- Step 2: identify reference descriptor ---
        default_ref = _REFERENCE_DESCRIPTOR[self.mode]
        applicable_names = {n for n, _ in applicable}
        if default_ref in applicable_names:
            ref_name = default_ref
        else:
            ref_name = max(solo_aucs, key=solo_aucs.__getitem__)
            logger.warning(
                f"Reference descriptor '{default_ref}' not applicable; "
                f"falling back to '{ref_name}' (highest solo AUC) as reference"
            )

        ref_desc = next(d for n, d in applicable if n == ref_name)
        ref_auc = solo_aucs[ref_name]
        logger.info(f"  Reference descriptor: '{ref_name}'  solo AUC = {ref_auc:.4f}")

        # --- Step 3: initialize ensemble with reference ---
        passing = [(ref_name, ref_desc)]
        ensemble_oof = oof_preds[ref_name].copy()
        ensemble_auc = float(roc_auc_score(y_batch, ensemble_oof))
        logger.info(
            f"  Ensemble initialized with '{ref_name}'  AUC = {ensemble_auc:.4f}"
        )

        # --- Step 4: greedy forward selection (solo AUC descending) ---
        candidates = sorted(
            [(n, d) for n, d in applicable if n != ref_name],
            key=lambda nd: solo_aucs[nd[0]],
            reverse=True,
        )
        for name, desc in candidates:
            if len(passing) >= _MAX_DESCRIPTORS:
                logger.info(
                    f"  [{name}] SKIPPED — ensemble cap of {_MAX_DESCRIPTORS} reached"
                )
                continue
            gate_a = solo_aucs[name] >= ref_auc  # equal or better than reference
            candidate_oof = np.mean(
                [oof_preds[n] for n, _ in passing] + [oof_preds[name]], axis=0
            )
            candidate_ens_auc = float(roc_auc_score(y_batch, candidate_oof))
            gate_b = candidate_ens_auc >= ensemble_auc  # doesn't degrade ensemble

            if gate_a or gate_b:
                prev_ensemble_auc = ensemble_auc
                passing.append((name, desc))
                ensemble_oof = np.mean([oof_preds[n] for n, _ in passing], axis=0)
                ensemble_auc = float(roc_auc_score(y_batch, ensemble_oof))
                logger.info(
                    f"  [{name}] ADDED   gate_A={gate_a} "
                    f"(solo={solo_aucs[name]:.4f} >= ref={ref_auc:.4f})  "
                    f"gate_B={gate_b} "
                    f"(ens={candidate_ens_auc:.4f} >= prev={prev_ensemble_auc:.4f})  "
                    f"→ ensemble AUC now {ensemble_auc:.4f}"
                )
            else:
                logger.info(
                    f"  [{name}] DROPPED gate_A={gate_a} "
                    f"(solo={solo_aucs[name]:.4f} < ref={ref_auc:.4f})  "
                    f"gate_B={gate_b} "
                    f"(ens would be {candidate_ens_auc:.4f} < current={ensemble_auc:.4f})"
                )

        logger.info(
            f"  Final ensemble: {[n for n, _ in passing]}  AUC = {ensemble_auc:.4f}"
        )

        # --- Step 5: summary table ---
        passing_names = {n for n, _ in passing}
        screen_rows = []
        for name, _ in applicable:
            if name == ref_name:
                status = "reference"
            elif name in passing_names:
                status = "pass"
            else:
                status = "drop"
            screen_rows.append(
                {
                    "name": name,
                    "n_features": int(feature_map[name].shape[1]),
                    "proxy_auc": solo_aucs[name],
                    "status": status,
                }
            )
        logger.proxy_screen_table(screen_rows)

        logger.rule()
        return [
            (name, desc, feature_map[name], solo_aucs[name]) for name, desc in passing
        ]

    def _proxy_screen_regression(
        self,
        smiles_list: list,
        y: np.ndarray,
        applicable: list,
        precomputed: dict = None,
    ) -> list:
        """Regression twin of :meth:`_proxy_screen`: same greedy rule, Spearman ρ.

        A shallow random-forest regressor on 3 quantile-stratified folds gives each
        descriptor an out-of-fold prediction. The reference descriptor seeds the ensemble;
        each other descriptor, best first, is added when its solo ρ matches the
        reference's (gate A) or adding it does not raise the averaged ensemble's RMSE
        (gate B; RMSE, the metric every regression head is selected on), up to the
        descriptor cap. No class-balance batching applies.

        A descriptor whose out-of-fold ρ is undefined (constant predictions) ranks last
        and is reported with an unknown proxy skill (``None``), never as a number the
        ensemble weighting would read as real.
        """
        from ..preprocessors import Preprocessor
        from ..utils.metrics import spearman_score
        from ..utils.splits import quantile_bins
        from joblib import parallel_backend
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        def rho(pred):
            # For ranking only: an undefined ρ sorts below every real one.
            value = spearman_score(y, pred)
            return -1.0 if value is None else value

        def rmse(pred):
            return float(np.sqrt(np.mean((np.asarray(pred) - y) ** 2)))

        # Single-threaded on purpose: parallel prediction sums the trees in a varying
        # order (~1e-16), which can flip a tie in the screen between runs.
        proxy_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            max_features=1.0 / 3.0,  # the regression forest's own split rule
            n_jobs=1,
            random_state=42,
        )
        bins = quantile_bins(y, max(1, min(5, len(y) // 6)))
        cv = list(
            StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(
                np.zeros(len(y)), bins
            )
        )

        logger.rule("Descriptor screening — proxy CV (regression)")
        logger.info(f"  Using full dataset ({len(y):,} samples) for proxy CV")
        oof_preds, solo_rhos, feature_map = {}, {}, {}
        for name, desc in applicable:
            X_full = (precomputed or {}).get(name)
            if X_full is None:
                logger.info(
                    f"  [{name}] Computing features for {len(smiles_list):,} SMILES..."
                )
                X_full = desc.transform(smiles_list)
            feature_map[name] = X_full
            prep = Preprocessor("regression")
            prep.fit(X_full, y)
            X_prep = prep.transform(X_full)
            with parallel_backend("threading", n_jobs=3):
                pred = cross_val_predict(proxy_model, X_prep, y, cv=cv, n_jobs=3)
            oof_preds[name] = pred
            solo_rhos[name] = rho(pred)
            logger.info(f"  [{name}] solo proxy ρ = {solo_rhos[name]:.4f}")

        default_ref = _REFERENCE_DESCRIPTOR[self.mode]
        if default_ref in solo_rhos:
            ref_name = default_ref
        else:
            ref_name = max(solo_rhos, key=solo_rhos.__getitem__)
            logger.warning(
                f"Reference descriptor '{default_ref}' not applicable; "
                f"falling back to '{ref_name}' (highest solo ρ) as reference"
            )
        ref_rho = solo_rhos[ref_name]
        passing = [(ref_name, next(d for n, d in applicable if n == ref_name))]
        ensemble_rmse = rmse(oof_preds[ref_name])

        candidates = sorted(
            [(n, d) for n, d in applicable if n != ref_name],
            key=lambda nd: solo_rhos[nd[0]],
            reverse=True,
        )
        for name, desc in candidates:
            if len(passing) >= _MAX_DESCRIPTORS:
                logger.info(
                    f"  [{name}] SKIPPED — ensemble cap of {_MAX_DESCRIPTORS} reached"
                )
                continue
            gate_a = solo_rhos[name] >= ref_rho
            candidate_rmse = rmse(
                np.mean([oof_preds[n] for n, _ in passing] + [oof_preds[name]], axis=0)
            )
            gate_b = candidate_rmse <= ensemble_rmse
            if gate_a or gate_b:
                passing.append((name, desc))
                ensemble_rmse = rmse(
                    np.mean([oof_preds[n] for n, _ in passing], axis=0)
                )
                logger.info(
                    f"  [{name}] ADDED   gate_A={gate_a} gate_B={gate_b} "
                    f"→ ensemble RMSE now {ensemble_rmse:.4f}"
                )
            else:
                logger.info(
                    f"  [{name}] DROPPED gate_A={gate_a} gate_B={gate_b} "
                    f"(ens RMSE would be {candidate_rmse:.4f} > {ensemble_rmse:.4f})"
                )

        passing_names = {n for n, _ in passing}
        logger.proxy_screen_table(
            [
                {
                    "name": name,
                    "n_features": int(feature_map[name].shape[1]),
                    "proxy_auc": solo_rhos[name],
                    "status": "reference"
                    if name == ref_name
                    else ("pass" if name in passing_names else "drop"),
                }
                for name, _ in applicable
            ],
            metric="ρ",
        )
        logger.rule()
        return [
            (name, desc, feature_map[name], spearman_score(y, oof_preds[name]))
            for name, desc in passing
        ]
