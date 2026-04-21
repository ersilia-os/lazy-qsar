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
}

DESCRIPTORS_MODE = {
    "default": ["chemeleon", "rdkit", "cddd"],
    "fast": ["rdkit", "morgan"],
    "slow": ["chemeleon", "morgan", "rdkit", "cddd"],
}

DESCRIPTORS_MODE = {k: sorted(v) for k, v in DESCRIPTORS_MODE.items()}


def get_descriptor_type(descriptor_name):
    module_name, class_name = DESCRIPTOR_TYPES[descriptor_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def _softmax_weights(A: np.ndarray) -> np.ndarray:
    """Row-wise softmax of a (B, D) matrix."""
    A = A - A.max(axis=1, keepdims=True)
    E = np.exp(A)
    return E / E.sum(axis=1, keepdims=True)


class ArtifactWrapper(object):
    def __init__(self, descriptors, artifacts, ad_artifacts=None, cv_aucs=None,
                 active_descriptors=None, ad_hard_cutoffs=None):
        self.descriptors = descriptors
        self.artifacts = artifacts
        self.ad_artifacts = ad_artifacts
        self.cv_aucs = cv_aucs
        self.active_descriptors = active_descriptors  # list[bool] or None
        self.ad_hard_cutoffs = ad_hard_cutoffs        # list[float] or None

    def predict_proba(self, smiles_list):
        active_mask = self.active_descriptors or [True] * len(self.descriptors)
        active_indices = [i for i, a in enumerate(active_mask) if a]
        if not active_indices:
            active_indices = list(range(len(self.descriptors)))

        y_hats = []
        ad_scores = []
        for i in active_indices:
            X = self.descriptors[i].transform(smiles_list)
            y_hats.append(np.array(self.artifacts[i].predict_proba(X))[:, 1])
            if self.ad_artifacts is not None and self.ad_artifacts[i] is not None:
                ad_scores.append(self.ad_artifacts[i].score(X))

        Y = np.stack(y_hats, axis=1).astype(np.float64)  # (B, D_active)

        if len(ad_scores) == len(active_indices):
            A = np.stack(ad_scores, axis=1).astype(np.float64)
            W = np.ones((len(y_hats[0]), len(active_indices)), dtype=np.float64)
            if self.ad_hard_cutoffs is not None:
                for j, i in enumerate(active_indices):
                    W[A[:, j] < self.ad_hard_cutoffs[i], j] = 0.0
            all_ood = W.sum(axis=1) == 0
            if all_ood.any():
                W[all_ood] = 1.0
            W /= W.sum(axis=1, keepdims=True)
            logits = np.log(np.clip(Y, 1e-7, 1 - 1e-7) / np.clip(1 - Y, 1e-7, 1 - 1e-7))
            avg_logit = (W * logits).sum(axis=1)
            y_hat_1 = 1.0 / (1.0 + np.exp(-avg_logit))
        else:
            y_hat_1 = Y.mean(axis=1)
        return np.vstack((1 - y_hat_1, y_hat_1)).T

    def predict(self, smiles_list, cutoff=0.5):
        y_hat = self.predict_proba(smiles_list)[:, 1]
        return (y_hat >= cutoff).astype(int)


class LazyClassifierQSAR(object):
    def __init__(
        self,
        mode: str = "default",
    ):
        assert mode in ["default", "fast", "slow"], (
            f"Mode {mode} not recognized. Choose from 'default', 'fast', or 'slow'."
        )

        descriptor_types = DESCRIPTORS_MODE[mode]

        self.mode = mode
        self.descriptor_types = descriptor_types
        self.descriptors = [
            get_descriptor_type(descriptor_type)() for descriptor_type in descriptor_types
        ]
        self.is_saved = False
        self._feature_cache = {}  # {(descriptor_idx, smiles_hash): X}
        self._n_train_samples = None

    def _smiles_hash(self, smiles_list):
        return hashlib.md5("\x00".join(smiles_list).encode()).hexdigest()

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

        y = np.array(y, dtype=int)
        n = len(smiles_list)
        pos_rate = float(y.mean())

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
                "oof_auc": oof_auc,
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

    def predict_proba(self, smiles_list):
        active_mask = getattr(self, "active_descriptors_", [True] * len(self.descriptor_types))
        ad_hard_cutoffs = getattr(self, "ad_hard_cutoffs_", None)
        quality_aucs = getattr(self, "quality_aucs_", None)

        active_indices = [i for i, a in enumerate(active_mask) if a]
        if not active_indices:
            active_indices = list(range(len(self.descriptor_types)))

        y_hats = []
        ad_scores = []
        for i in active_indices:
            X = self._transform_cached(i, smiles_list)
            y_hats.append(self.models[i].predict_proba(X=X)[:, 1])
            if self.ad_models:
                ad_scores.append(self.ad_models[i].score(X))

        Y = np.stack(y_hats, axis=1).astype(np.float64)  # (B, D_active)

        if len(ad_scores) == len(active_indices):
            A = np.stack(ad_scores, axis=1).astype(np.float64)  # (B, D_active)
            # Equal weights with hard AD veto
            W = np.ones((len(smiles_list), len(active_indices)), dtype=np.float64)
            if ad_hard_cutoffs is not None:
                for j, i in enumerate(active_indices):
                    W[A[:, j] < ad_hard_cutoffs[i], j] = 0.0

            # All-OOD fallback: equal weights
            all_ood = W.sum(axis=1) == 0
            if all_ood.any():
                W[all_ood] = 1.0

            W /= W.sum(axis=1, keepdims=True)  # (B, D_active)

            # Average in logit space → naturally upweights confident predictions
            logits = np.log(np.clip(Y, 1e-7, 1 - 1e-7) / np.clip(1 - Y, 1e-7, 1 - 1e-7))
            avg_logit = (W * logits).sum(axis=1)
            y_hat_1 = 1.0 / (1.0 + np.exp(-avg_logit))

            winner = W.argmax(axis=1)
            oof_aucs = getattr(self, "oof_aucs_", None)
            rows = []
            for j, i in enumerate(active_indices):
                name = self.descriptor_types[i]
                rows.append({
                    "name":        name,
                    "oof_auc":     float(oof_aucs[i]) if oof_aucs else float("nan"),
                    "ad_mean":     float(A[:, j].mean()),
                    "ad_std":      float(A[:, j].std()),
                    "ad_min":      float(A[:, j].min()),
                    "ad_max":      float(A[:, j].max()),
                    "weight_mean": float(W[:, j].mean()),
                    "weight_std":  float(W[:, j].std()),
                    "wins":        int((winner == j).sum()),
                    "pred_mean":   float(Y[:, j].mean()),
                })
            logger.ad_weights_table(rows, n_samples=len(smiles_list))
        else:
            y_hat_1 = Y.mean(axis=1)
        return np.vstack((1 - y_hat_1, y_hat_1)).T

    def predict(self, smiles_list, threshold=0.5):
        return (self.predict_proba(smiles_list)[:, 1] >= threshold).astype(int)

    def save_raw(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        meta = {
            "descriptor_types": self.descriptor_types,
            "oof_aucs": {
                name: float(auc)
                for name, auc in zip(self.descriptor_types, self.oof_aucs_)
            } if hasattr(self, "oof_aucs_") else {},
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
        mode = None
        for k, v in DESCRIPTORS_MODE.items():
            if set(v) == set(descriptor_types):
                mode = k
                break
        if mode is None:
            raise Exception(
                "Could not infer mode from descriptor types found in the model directory."
            )
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
        obj.descriptors = descriptors
        obj.models = models
        obj.ad_models = ad_models if any(a is not None for a in ad_models) else []
        obj._n_train_samples = None
        obj.is_saved = True
        meta_path = os.path.join(model_dir, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            oof_map = meta.get("oof_aucs", {})
            train_map = meta.get("train_aucs", {})
            quality_map = meta.get("quality_aucs", {})
            active_map = meta.get("active_descriptors", {})
            cutoff_map = meta.get("ad_hard_cutoffs", {})
            obj.oof_aucs_ = [oof_map.get(d, 1.0) for d in descriptor_types]
            obj.train_aucs_ = [train_map.get(d, 0.0) for d in descriptor_types]
            obj.quality_aucs_ = [quality_map.get(d, oof_map.get(d, 1.0)) for d in descriptor_types]
            obj.active_descriptors_ = [active_map.get(d, True) for d in descriptor_types]
            obj.ad_hard_cutoffs_ = [cutoff_map.get(d, 0.0) for d in descriptor_types] if cutoff_map else None
        else:
            obj.oof_aucs_ = [1.0] * len(descriptor_types)
            obj.train_aucs_ = [0.0] * len(descriptor_types)
            obj.quality_aucs_ = [1.0] * len(descriptor_types)
            obj.active_descriptors_ = [True] * len(descriptor_types)
            obj.ad_hard_cutoffs_ = None
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
        cv_aucs = None
        active_descriptors = None
        ad_hard_cutoffs = None
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            oof_map = meta.get("oof_aucs", {})
            quality_map = meta.get("quality_aucs", {})
            active_map = meta.get("active_descriptors", {})
            cutoff_map = meta.get("ad_hard_cutoffs", {})
            cv_aucs = [quality_map.get(d, oof_map.get(d, 1.0)) for d in descriptor_types]
            active_descriptors = [active_map.get(d, True) for d in descriptor_types] if active_map else None
            ad_hard_cutoffs = [cutoff_map.get(d, 0.0) for d in descriptor_types] if cutoff_map else None
        return ArtifactWrapper(
            descriptors=descriptors,
            artifacts=artifacts,
            ad_artifacts=ad_artifacts if has_ad else None,
            cv_aucs=cv_aucs,
            active_descriptors=active_descriptors,
            ad_hard_cutoffs=ad_hard_cutoffs,
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
