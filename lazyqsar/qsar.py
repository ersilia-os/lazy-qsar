import hashlib
import json
import os
import shutil
import numpy as np

from .utils.logging import logger


DESCRIPTOR_TYPES = {
    "chemeleon": ("lazyqsar.descriptors.chemeleon", "ChemeleonDescriptor"),
    "morgan": ("lazyqsar.descriptors.morgan", "MorganFingerprint"),
    "rdkit": ("lazyqsar.descriptors.rdkit_descriptors", "RDKitDescriptor"),
    "cddd": ("lazyqsar.descriptors.cddd", "ContinuousDataDrivenDescriptor"),
    "clamp": ("lazyqsar.descriptors.clamp", "ClampDescriptor"),
}

DESCRIPTORS_MODE = {
    "fast": ["morgan"],
    "slow": ["chemeleon", "morgan", "rdkit", "cddd", "clamp"],
}

DESCRIPTORS_MODE = {k: sorted(v) for k, v in DESCRIPTORS_MODE.items()}


def get_descriptor_type(descriptor_name):
    module_name, class_name = DESCRIPTOR_TYPES[descriptor_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def _build_weight_matrix(Y, R, A, oof_aucs, proxy_aucs, rank_error_curves,
                         active_indices, ad_hard_cutoffs):
    """Compute normalized weight matrix W (B, D_active).

    Parameters
    ----------
    Y  : (B, D) calibrated probabilities per descriptor
    R  : (B, D) quantile ranks per descriptor, or None
    A  : (B, D) AD scores per descriptor, or None when no AD models
    oof_aucs, proxy_aucs : list indexed by full descriptor position
    rank_error_curves    : list indexed by full descriptor position
    active_indices       : indices into the full descriptor list
    ad_hard_cutoffs      : list indexed by full descriptor position, or None

    Returns
    -------
    W    : (B, D) normalized weight matrix
    base : (D,) global AUC skill scores (for logging / all-OOD fallback)
    """
    B, D = Y.shape

    # Global base: max(0, mean(oof_auc, proxy_auc) − 0.5)
    base_scores = []
    for i in active_indices:
        vals = []
        if oof_aucs and oof_aucs[i] is not None:
            vals.append(float(oof_aucs[i]))
        if proxy_aucs and proxy_aucs[i] is not None:
            vals.append(float(proxy_aucs[i]))
        base_scores.append(max(0.0, float(np.mean(vals)) - 0.5) if vals else 0.0)
    base = np.array(base_scores, dtype=np.float64)
    if base.sum() == 0:
        base = np.ones(D, dtype=np.float64)

    if A is not None:
        # Per-sample reliability from rank→error curves, or |rank−0.5|×2 fallback
        if R is not None:
            if (rank_error_curves
                    and all(rank_error_curves[i] is not None for i in active_indices)):
                reliability = np.zeros((B, D), dtype=np.float64)
                for j, i in enumerate(active_indices):
                    r_knots, e_knots = rank_error_curves[i]
                    reliability[:, j] = 1.0 - np.interp(R[:, j], r_knots, e_knots)
            else:
                reliability = np.abs(R - 0.5) * 2
            W = 0.5 * base[np.newaxis, :] + 0.5 * reliability
        else:
            W = np.tile(base, (B, 1))

        # AD hard-cutoff veto
        if ad_hard_cutoffs is not None:
            for j, i in enumerate(active_indices):
                W[A[:, j] < ad_hard_cutoffs[i], j] = 0.0

        # All-OOD fallback: restore AUC-based base weights
        all_ood = W.sum(axis=1) == 0
        if all_ood.any():
            W[all_ood] = base

        W /= W.sum(axis=1, keepdims=True)
    else:
        W = np.full((B, D), 1.0 / D, dtype=np.float64)

    return W, base


def _smiles_md5(smiles_list):
    return hashlib.md5("\x00".join(smiles_list).encode()).hexdigest()


class ArtifactWrapper(object):
    def __init__(self, descriptors, artifacts, ad_artifacts=None,
                 active_descriptors=None, ad_hard_cutoffs=None,
                 oof_aucs=None, proxy_aucs=None,
                 rank_error_curves=None, population_prior=0.5,
                 descriptor_types=None):
        self.descriptors = descriptors
        self.artifacts = artifacts
        self.ad_artifacts = ad_artifacts
        self.active_descriptors = active_descriptors  # list[bool] or None
        self.ad_hard_cutoffs = ad_hard_cutoffs        # list[float] or None
        self.oof_aucs = oof_aucs                      # list[float|None]
        self.proxy_aucs = proxy_aucs                  # list[float|None]
        self.rank_error_curves = rank_error_curves    # list[(r_knots, e_knots)|None]
        self.population_prior = population_prior
        self.descriptor_types = descriptor_types      # list[str] or None
        self._ensemble_cache = {}

    def _compute_ensemble(self, smiles_list):
        cache_key = _smiles_md5(smiles_list)
        if cache_key in self._ensemble_cache:
            return self._ensemble_cache[cache_key]

        active_mask = self.active_descriptors or [True] * len(self.descriptors)
        active_indices = [i for i, a in enumerate(active_mask) if a]
        if not active_indices:
            active_indices = list(range(len(self.descriptors)))

        y_hats, score_preds, rank_preds, ad_scores = [], [], [], []
        for i in active_indices:
            X = self.descriptors[i].transform(smiles_list)
            y_hats.append(np.array(self.artifacts[i].predict_proba(X))[:, 1])
            try:
                score_preds.append(self.artifacts[i].predict_score(X)[:, 1])
            except Exception:
                score_preds.append(None)
            try:
                rank_preds.append(self.artifacts[i].predict_rank(X)[:, 1])
            except Exception:
                rank_preds.append(None)
            if self.ad_artifacts is not None and self.ad_artifacts[i] is not None:
                ad_scores.append(self.ad_artifacts[i].score(X))

        B = len(smiles_list)
        D = len(active_indices)
        Y = np.stack(y_hats, axis=1).astype(np.float64)
        R = np.stack(rank_preds, axis=1).astype(np.float64) if all(
            r is not None for r in rank_preds) else None
        S = np.stack(score_preds, axis=1).astype(np.float64) if all(
            s is not None for s in score_preds) else Y.copy()
        A = np.stack(ad_scores, axis=1).astype(np.float64) if len(
            ad_scores) == D else None

        W, base = _build_weight_matrix(
            Y, R, A,
            self.oof_aucs, self.proxy_aucs, self.rank_error_curves,
            active_indices, self.ad_hard_cutoffs,
        )

        if A is not None:
            names = self.descriptor_types or [str(i) for i in range(len(self.descriptors))]
            rows = []
            for j, i in enumerate(active_indices):
                rows.append({
                    "name":        names[i],
                    "oof_auc":     float(self.oof_aucs[i]) if self.oof_aucs else float("nan"),
                    "proxy_auc":   float(self.proxy_aucs[i]) if (
                        self.proxy_aucs and self.proxy_aucs[i] is not None) else None,
                    "ad_mean":     float(A[:, j].mean()),
                    "ad_std":      float(A[:, j].std()),
                    "ad_min":      float(A[:, j].min()),
                    "ad_max":      float(A[:, j].max()),
                    "weight_mean": float(W[:, j].mean()),
                    "weight_std":  float(W[:, j].std()),
                    "vetoed":      int((W[:, j] == 0).sum()),
                    "pred_mean":   float(Y[:, j].mean()),
                })
            logger.ad_weights_table(rows, n_samples=B)

        if R is None:
            R = np.full((B, D), 0.5, dtype=np.float64)

        result = (W, Y, R, S, active_indices)
        self._ensemble_cache[cache_key] = result
        return result

    def predict_proba(self, smiles_list):
        W, Y, R, S, _ = self._compute_ensemble(smiles_list)
        logits = np.log(np.clip(Y, 1e-7, 1 - 1e-7) / np.clip(1 - Y, 1e-7, 1 - 1e-7))
        p1 = 1.0 / (1.0 + np.exp(-(W * logits).sum(axis=1)))
        return np.vstack((1 - p1, p1)).T

    def predict_logit(self, smiles_list):
        W, Y, R, S, _ = self._compute_ensemble(smiles_list)
        logits = np.log(np.clip(Y, 1e-7, 1 - 1e-7) / np.clip(1 - Y, 1e-7, 1 - 1e-7))
        l1 = (W * logits).sum(axis=1)
        return np.vstack((-l1, l1)).T

    def predict_rank(self, smiles_list):
        W, Y, R, S, _ = self._compute_ensemble(smiles_list)
        r1 = (W * R).sum(axis=1)
        return np.vstack((1 - r1, r1)).T

    def predict_score(self, smiles_list):
        W, Y, R, S, _ = self._compute_ensemble(smiles_list)
        s1 = (W * S).sum(axis=1)
        return np.vstack((1 - s1, s1)).T

    def predict_lift(self, smiles_list):
        prior = self.population_prior
        proba = self.predict_proba(smiles_list)
        return np.column_stack([proba[:, 0] / max(1 - prior, 1e-7),
                                proba[:, 1] / max(prior, 1e-7)])

    def predict(self, smiles_list, cutoff=0.5):
        return (self.predict_proba(smiles_list)[:, 1] >= cutoff).astype(int)


class LazyClassifierQSAR(object):
    def __init__(
        self,
        mode: str = "slow",
    ):
        assert mode in ("fast", "slow"), (
            f"Mode '{mode}' not recognized. Choose from 'fast' or 'slow'."
        )
        self.mode = mode
        self.descriptor_types = DESCRIPTORS_MODE[mode]
        self.descriptors = []  # populated in fit() after applicability check
        self.is_saved = False
        self._feature_cache = {}
        self._ensemble_cache = {}

    def _smiles_hash(self, smiles_list):
        return _smiles_md5(smiles_list)

    def _transform_cached(self, i, smiles_list):
        key = (i, self._smiles_hash(smiles_list))
        if key not in self._feature_cache:
            self._feature_cache[key] = self.descriptors[i].transform(smiles_list)
        else:
            logger.debug(f"Using cached features for descriptor: {self.descriptor_types[i]}")
        return self._feature_cache[key]

    def fit(self, smiles_list, y):
        import time
        from .agnostic import LazyClassifier
        from .applicability import ApplicabilityDomain
        from .descriptors.portfolio import DescriptorPortfolio

        # Clear any cached state from a previous fit.
        self._feature_cache.clear()
        self._ensemble_cache.clear()

        y = np.array(y, dtype=int)
        n = len(smiles_list)
        pos_rate = float(y.mean())
        self.population_prior_ = pos_rate

        applicable = DescriptorPortfolio(self.mode).select(smiles_list, y=y)
        self.descriptor_types = [name for name, _, _, _ in applicable]
        self.descriptors      = [desc for _, desc, _, _ in applicable]
        self.proxy_aucs_      = [pauc for _, _, _, pauc in applicable]

        # Pre-populate feature cache with matrices computed during screening.
        smiles_hash = self._smiles_hash(smiles_list)
        for i, (_, _, X, _) in enumerate(applicable):
            if X is not None:
                self._feature_cache[(i, smiles_hash)] = X

        logger.rule("LazyClassifierQSAR")
        logger.info(
            f"mode={self.mode}  descriptors={self.descriptor_types}  "
            f"n={n:,}  pos_rate={pos_rate:.1%}"
        )

        self.models = []
        self.ad_models = []
        self.oof_aucs_ = []
        self.train_aucs_ = []
        self.quality_aucs_ = []
        self._rank_error_curves_ = []
        _ad_hard_cutoffs_raw = []
        desc_rows = []

        for i, desc_name in enumerate(self.descriptor_types):
            t0 = time.perf_counter()
            X = self._transform_cached(i, smiles_list)
            feat_time = time.perf_counter() - t0

            sparsity = float((X == 0).mean())
            logger.info(
                f"[{desc_name}] p={X.shape[1]:,}  sparsity={sparsity:.3f}  "
                f"feat_time={feat_time:.1f}s"
            )

            model = LazyClassifier()
            model.fit(X=X, y=y)
            self.models.append(model)

            # Build rank→error curve from training predictions (20 knots, windowed).
            # Training-set predictions are optimistic but sufficient as a relative
            # reliability signal: high |rank - 0.5| should map to low error.
            try:
                _p = model.predict_proba(X=X)[:, 1]
                _r = model.predict_rank(X=X)[:, 1]
                _err = np.abs(_p - y.astype(float))
                _sidx = np.argsort(_r)
                _rs, _es = _r[_sidx], _err[_sidx]
                _nk = min(20, len(_rs))
                _ki = np.round(np.linspace(0, len(_rs) - 1, _nk)).astype(int)
                _hw = max(1, len(_rs) // (_nk * 2))
                _r_knots = _rs[_ki]
                _e_knots = np.array([_es[max(0, k - _hw):k + _hw + 1].mean() for k in _ki])
                self._rank_error_curves_.append((_r_knots.tolist(), _e_knots.tolist()))
            except Exception:
                self._rank_error_curves_.append(None)

            oof_auc = model.oof_auc_
            train_auc = model.train_auc_
            gap = train_auc - oof_auc
            quality = oof_auc - gap   # α=1: quality = 2*oof - train

            self.oof_aucs_.append(oof_auc)
            self.train_aucs_.append(train_auc)
            self.quality_aucs_.append(quality)

            ad = ApplicabilityDomain()
            ad.fit(X)
            self.ad_models.append(ad)
            train_ad = ad.score(X)
            _ad_hard_cutoffs_raw.append(float(np.percentile(train_ad, 5)))

            logger.info(
                f"[{desc_name}] OOF={oof_auc:.4f}  train={train_auc:.4f}  "
                f"gap={gap:.4f}  quality={quality:.4f}  "
                f"AD comps={ad.pca_.n_components_}"
            )

            desc_rows.append({
                "name": desc_name,
                "n_features": X.shape[1],
                "sparsity": sparsity,
                "feat_time": feat_time,
                "ad_n_components": ad.pca_.n_components_,
                "ad_cal_min": float(ad.cal_knots_[0]),
                "ad_cal_max": float(ad.cal_knots_[-1]),
                "proxy_auc": self.proxy_aucs_[i],
                "train_auc": train_auc,
                "quality_auc": quality,
            })

        # Descriptor-level pruning: drop if OOF AUC < floor OR < best - gap
        best_oof = max(self.oof_aucs_)
        _floor, _gap = 0.55, 0.10
        active_mask = [
            (auc >= _floor) and (auc >= best_oof - _gap)
            for auc in self.oof_aucs_
        ]
        if not any(active_mask):
            active_mask = [True] * len(self.oof_aucs_)
        self.active_descriptors_ = active_mask
        self.ad_hard_cutoffs_ = _ad_hard_cutoffs_raw

        for row, active in zip(desc_rows, active_mask):
            row["active"] = active

        logger.rule()
        logger.descriptor_table(desc_rows)

    def _compute_ensemble(self, smiles_list):
        """Compute (W, Y, R, S, active_indices) for smiles_list, cached by SMILES hash."""
        cache_key = self._smiles_hash(smiles_list)
        if cache_key in self._ensemble_cache:
            return self._ensemble_cache[cache_key]

        active_mask = getattr(self, "active_descriptors_", [True] * len(self.descriptor_types))
        ad_hard_cutoffs = getattr(self, "ad_hard_cutoffs_", None)

        active_indices = [i for i, a in enumerate(active_mask) if a]
        if not active_indices:
            active_indices = list(range(len(self.descriptor_types)))

        y_hats, score_preds, rank_preds, ad_scores = [], [], [], []
        for i in active_indices:
            X = self._transform_cached(i, smiles_list)
            y_hats.append(self.models[i].predict_proba(X=X)[:, 1])
            try:
                score_preds.append(self.models[i].predict_score(X=X)[:, 1])
            except Exception:
                score_preds.append(None)
            if self.ad_models:
                ad_scores.append(self.ad_models[i].score(X))
            try:
                rank_preds.append(self.models[i].predict_rank(X=X)[:, 1])
            except Exception:
                rank_preds.append(None)

        B = len(smiles_list)
        D = len(active_indices)
        Y = np.stack(y_hats, axis=1).astype(np.float64)
        R = np.stack(rank_preds, axis=1).astype(np.float64) if all(
            r is not None for r in rank_preds) else None
        S = np.stack(score_preds, axis=1).astype(np.float64) if all(
            s is not None for s in score_preds) else Y.copy()
        A = np.stack(ad_scores, axis=1).astype(np.float64) if len(
            ad_scores) == D else None

        W, base = _build_weight_matrix(
            Y, R, A,
            getattr(self, "oof_aucs_", None),
            getattr(self, "proxy_aucs_", None),
            getattr(self, "_rank_error_curves_", None),
            active_indices, ad_hard_cutoffs,
        )

        if A is not None:
            oof_aucs   = getattr(self, "oof_aucs_", None)
            proxy_aucs = getattr(self, "proxy_aucs_", None)
            rows = []
            for j, i in enumerate(active_indices):
                rows.append({
                    "name":        self.descriptor_types[i],
                    "oof_auc":     float(oof_aucs[i]) if oof_aucs else float("nan"),
                    "proxy_auc":   float(proxy_aucs[i]) if (
                        proxy_aucs and proxy_aucs[i] is not None) else None,
                    "ad_mean":     float(A[:, j].mean()),
                    "ad_std":      float(A[:, j].std()),
                    "ad_min":      float(A[:, j].min()),
                    "ad_max":      float(A[:, j].max()),
                    "weight_mean": float(W[:, j].mean()),
                    "weight_std":  float(W[:, j].std()),
                    "vetoed":      int((W[:, j] == 0).sum()),
                    "pred_mean":   float(Y[:, j].mean()),
                })
            logger.ad_weights_table(rows, n_samples=B)

        if R is None:
            R = np.full((B, D), 0.5, dtype=np.float64)

        result = (W, Y, R, S, active_indices)
        self._ensemble_cache[cache_key] = result
        return result

    def predict_proba(self, smiles_list):
        W, Y, R, S, _ = self._compute_ensemble(smiles_list)
        logits = np.log(np.clip(Y, 1e-7, 1 - 1e-7) / np.clip(1 - Y, 1e-7, 1 - 1e-7))
        p1 = 1.0 / (1.0 + np.exp(-(W * logits).sum(axis=1)))
        return np.vstack((1 - p1, p1)).T

    def predict_logit(self, smiles_list):
        W, Y, R, S, _ = self._compute_ensemble(smiles_list)
        logits = np.log(np.clip(Y, 1e-7, 1 - 1e-7) / np.clip(1 - Y, 1e-7, 1 - 1e-7))
        l1 = (W * logits).sum(axis=1)
        return np.vstack((-l1, l1)).T

    def predict_rank(self, smiles_list):
        W, Y, R, S, _ = self._compute_ensemble(smiles_list)
        r1 = (W * R).sum(axis=1)
        return np.vstack((1 - r1, r1)).T

    def predict_score(self, smiles_list):
        W, Y, R, S, _ = self._compute_ensemble(smiles_list)
        s1 = (W * S).sum(axis=1)
        return np.vstack((1 - s1, s1)).T

    def predict_lift(self, smiles_list):
        prior = getattr(self, "population_prior_", 0.5)
        proba = self.predict_proba(smiles_list)
        return np.column_stack([proba[:, 0] / max(1 - prior, 1e-7),
                                proba[:, 1] / max(prior, 1e-7)])

    def predict(self, smiles_list, threshold=0.5):
        return (self.predict_proba(smiles_list)[:, 1] >= threshold).astype(int)

    def save_raw(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        meta = {
            "mode": self.mode,
            "descriptor_types": self.descriptor_types,
            "population_prior": float(self.population_prior_) if hasattr(self, "population_prior_") else 0.5,
            "oof_aucs": {
                name: float(auc)
                for name, auc in zip(self.descriptor_types, self.oof_aucs_)
            } if hasattr(self, "oof_aucs_") else {},
            "proxy_aucs": {
                name: float(auc)
                for name, auc in zip(self.descriptor_types, self.proxy_aucs_)
                if auc is not None
            } if hasattr(self, "proxy_aucs_") else {},
            "train_aucs": {
                name: float(auc)
                for name, auc in zip(self.descriptor_types, self.train_aucs_)
            } if hasattr(self, "train_aucs_") else {},
            "quality_aucs": {
                name: float(auc)
                for name, auc in zip(self.descriptor_types, self.quality_aucs_)
            } if hasattr(self, "quality_aucs_") else {},
            "active_descriptors": {
                name: bool(active)
                for name, active in zip(self.descriptor_types, self.active_descriptors_)
            } if hasattr(self, "active_descriptors_") else {},
            "ad_hard_cutoffs": {
                name: float(c)
                for name, c in zip(self.descriptor_types, self.ad_hard_cutoffs_)
            } if hasattr(self, "ad_hard_cutoffs_") else {},
            "rank_error_curves": {
                name: curve
                for name, curve in zip(self.descriptor_types, self._rank_error_curves_)
                if curve is not None
            } if hasattr(self, "_rank_error_curves_") else {},
        }
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        for i, descriptor_name in enumerate(self.descriptor_types):
            model_subdir = os.path.join(model_dir, descriptor_name)
            os.makedirs(model_subdir, exist_ok=True)
            logger.debug(f"Saving model to {model_subdir}")
            self.models[i].save(model_subdir)
            logger.debug(f"Saving descriptor to {model_subdir}")
            self.descriptors[i].save(model_subdir)
            if self.ad_models:
                ad_subdir = os.path.join(model_subdir, "applicability_domain")
                self.ad_models[i].save(ad_subdir)
        self.is_saved = True

    @classmethod
    def load_raw(cls, model_dir: str):
        from .applicability import ApplicabilityDomainArtifact

        descriptor_types = []
        for fn in os.listdir(model_dir):
            if fn in DESCRIPTOR_TYPES.keys():
                descriptor_types += [fn]
        descriptor_types = sorted(descriptor_types)
        # Read mode from metadata if available; fall back to inference for old models.
        meta_path = os.path.join(model_dir, "metadata.json")
        mode = None
        if os.path.isfile(meta_path):
            with open(meta_path) as _f:
                _meta = json.load(_f)
            mode = _meta.get("mode")
        if mode is None:
            for k, v in DESCRIPTORS_MODE.items():
                if set(v) == set(descriptor_types):
                    mode = k
                    break
        if mode is None:
            mode = "slow"  # safe default for models saved before mode tracking
        from .agnostic import LazyClassifier

        descriptors = []
        models = []
        ad_models = []
        for descriptor_type in descriptor_types:
            model_subdir = os.path.join(model_dir, descriptor_type)
            if not os.path.exists(model_subdir):
                raise FileNotFoundError(
                    f"Descriptor directory {model_subdir} does not exist."
                )
            descriptors += [get_descriptor_type(descriptor_type).load(model_subdir)]
            models += [LazyClassifier.load(model_subdir)]
            ad_subdir = os.path.join(model_subdir, "applicability_domain")
            if os.path.isdir(ad_subdir):
                ad_models.append(ApplicabilityDomainArtifact.load(ad_subdir))
            else:
                ad_models.append(None)

        obj = cls(mode=mode)
        obj.descriptor_types = descriptor_types
        obj.descriptors = descriptors
        obj.models = models
        obj.ad_models = ad_models if any(a is not None for a in ad_models) else []
        obj.is_saved = True
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            oof_map    = meta.get("oof_aucs", {})
            proxy_map  = meta.get("proxy_aucs", {})
            train_map  = meta.get("train_aucs", {})
            quality_map = meta.get("quality_aucs", {})
            active_map = meta.get("active_descriptors", {})
            cutoff_map = meta.get("ad_hard_cutoffs", {})
            curve_map  = meta.get("rank_error_curves", {})
            obj.population_prior_    = float(meta.get("population_prior", 0.5))
            obj.oof_aucs_            = [oof_map.get(d, 1.0) for d in descriptor_types]
            obj.proxy_aucs_          = [proxy_map.get(d) for d in descriptor_types]
            obj.train_aucs_          = [train_map.get(d, 0.0) for d in descriptor_types]
            obj.quality_aucs_        = [quality_map.get(d, oof_map.get(d, 1.0)) for d in descriptor_types]
            obj.active_descriptors_  = [active_map.get(d, True) for d in descriptor_types]
            obj.ad_hard_cutoffs_     = [cutoff_map.get(d, 0.0) for d in descriptor_types] if cutoff_map else None
            obj._rank_error_curves_  = [
                (np.array(curve_map[d][0]), np.array(curve_map[d][1])) if d in curve_map else None
                for d in descriptor_types
            ]
        else:
            obj.population_prior_   = 0.5
            obj.oof_aucs_           = [1.0] * len(descriptor_types)
            obj.proxy_aucs_         = [None] * len(descriptor_types)
            obj.train_aucs_         = [0.0] * len(descriptor_types)
            obj.quality_aucs_       = [1.0] * len(descriptor_types)
            obj.active_descriptors_ = [True] * len(descriptor_types)
            obj.ad_hard_cutoffs_    = None
            obj._rank_error_curves_ = [None] * len(descriptor_types)
        return obj

    def save_onnx(self, model_dir: str, clean: bool = True):
        # ONNX is already written by save_raw() via LazyClassifier.save().
        pass

    @classmethod
    def load_onnx(cls, model_dir: str):
        from .applicability import ApplicabilityDomainArtifact

        descriptor_types = []
        for fn in os.listdir(model_dir):
            if fn in DESCRIPTOR_TYPES.keys():
                descriptor_types += [fn]
        descriptor_types = sorted(descriptor_types)
        from .agnostic import LazyClassifier

        descriptors = []
        artifacts = []
        ad_artifacts = []
        for descriptor_type in descriptor_types:
            model_subdir = os.path.join(model_dir, descriptor_type)
            if not os.path.exists(model_subdir):
                raise FileNotFoundError(
                    f"Descriptor directory {model_subdir} does not exist."
                )
            descriptors += [get_descriptor_type(descriptor_type).load(model_subdir)]
            artifacts += [LazyClassifier.load(model_subdir)]
            ad_subdir = os.path.join(model_subdir, "applicability_domain")
            if os.path.isdir(ad_subdir):
                ad_artifacts.append(ApplicabilityDomainArtifact.load(ad_subdir))
            else:
                ad_artifacts.append(None)

        has_ad = any(a is not None for a in ad_artifacts)
        meta_path = os.path.join(model_dir, "metadata.json")
        oof_aucs = None
        proxy_aucs = None
        active_descriptors = None
        ad_hard_cutoffs = None
        rank_error_curves = None
        population_prior = 0.5
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            oof_map    = meta.get("oof_aucs", {})
            proxy_map  = meta.get("proxy_aucs", {})
            quality_map = meta.get("quality_aucs", {})
            active_map = meta.get("active_descriptors", {})
            cutoff_map = meta.get("ad_hard_cutoffs", {})
            curve_map  = meta.get("rank_error_curves", {})
            population_prior   = float(meta.get("population_prior", 0.5))
            oof_aucs           = [quality_map.get(d, oof_map.get(d, 1.0)) for d in descriptor_types]
            proxy_aucs         = [proxy_map.get(d) for d in descriptor_types]
            active_descriptors = [active_map.get(d, True) for d in descriptor_types] if active_map else None
            ad_hard_cutoffs    = [cutoff_map.get(d, 0.0) for d in descriptor_types] if cutoff_map else None
            rank_error_curves  = [
                (np.array(curve_map[d][0]), np.array(curve_map[d][1])) if d in curve_map else None
                for d in descriptor_types
            ] if curve_map else None
        return ArtifactWrapper(
            descriptors=descriptors,
            artifacts=artifacts,
            ad_artifacts=ad_artifacts if has_ad else None,
            active_descriptors=active_descriptors,
            ad_hard_cutoffs=ad_hard_cutoffs,
            oof_aucs=oof_aucs,
            proxy_aucs=proxy_aucs,
            rank_error_curves=rank_error_curves,
            population_prior=population_prior,
            descriptor_types=descriptor_types,
        )

    def save(self, model_dir: str, onnx: bool = True):
        if model_dir.endswith(".zip"):
            zip = True
            model_dir = model_dir[:-4]
        else:
            zip = False
        self.save_raw(model_dir)
        if onnx:
            self.save_onnx(model_dir)
        if zip:
            shutil.make_archive(model_dir, "zip", model_dir)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            return model_dir + ".zip"
        return model_dir

    @classmethod
    def load(cls, model_dir: str):
        if model_dir.endswith(".zip"):
            zip = True
        else:
            zip = False
        if zip:
            base_dir = model_dir[:-4]
            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)
            shutil.unpack_archive(model_dir, base_dir)
            model_dir = base_dir
        descriptor_types = []
        for fn in os.listdir(model_dir):
            if fn in DESCRIPTOR_TYPES.keys():
                descriptor_types += [fn]
        descriptor_types = sorted(descriptor_types)
        for descriptor_type in descriptor_types:
            model_subdir = os.path.join(model_dir, descriptor_type)
            for fn in os.listdir(model_subdir):
                if fn.endswith(".onnx"):
                    return cls.load_onnx(model_dir=model_dir)
        obj = cls.load_raw(model_dir=model_dir)
        if zip:
            shutil.rmtree(base_dir)
        return obj


class LazyRegressorQSAR:
    """Placeholder — not yet implemented."""

    def __init__(self, mode: str = "default"):
        raise NotImplementedError("LazyRegressorQSAR is not yet implemented.")


class LazyQSAR:
    """
    Dispatcher that returns the appropriate QSAR class based on task.

    LazyQSAR(task='classification', **kwargs)  →  LazyClassifierQSAR(**kwargs)
    LazyQSAR(task='regression', **kwargs)      →  LazyRegressorQSAR(**kwargs)
    """

    def __new__(cls, task: str = "classification", **kwargs):
        if task == "classification":
            return LazyClassifierQSAR(**kwargs)
        elif task == "regression":
            return LazyRegressorQSAR(**kwargs)
        else:
            raise ValueError(
                f"Unknown task {task!r}. Choose 'classification' or 'regression'."
            )
