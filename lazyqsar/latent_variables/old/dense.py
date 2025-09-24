import numpy as np
import optuna
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.utils.validation import check_is_fitted

from ...utils.logging import logger
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


MIN_FEATURES = 5
MAX_FEATURES = 2048
NUM_TRIALS = 20


class KFeaturesReducer(object):

    def __init__(self, k_features):
        self.k_features = k_features

    def fit(self, X, y):
        self.var_filter_ = VarianceThreshold(threshold=0.0)
        self.var_filter_.fit(X)
        X_nonconst = self.var_filter_.transform(X)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_nonconst)
        k = min(self.k_features, X.shape[1])
        self.selector_ = SelectKBest(score_func=f_classif, k=k)
        self.selector_.fit(X_scaled, y)
        return self
    
    def transform(self, X):
        check_is_fitted(self, ["var_filter_", "scaler_", "selector_"])
        X_nonconst = self.var_filter_.transform(X)
        X_scaled = self.scaler_.transform(X_nonconst)
        X_selected = self.selector_.transform(X_scaled)
        return X_selected


def get_k_features_parameter(X, y):

    def dense_k_features_initial_limits(X, y):
        if X.shape[1] < MIN_FEATURES:
            return None, None
        var_filter_ = VarianceThreshold(threshold=0.0)
        var_filter_.fit(X)
        X = var_filter_.transform(X)
        if X.shape[1] < MIN_FEATURES:
            return None, None
        scaler_ = StandardScaler()
        X = scaler_.fit_transform(X)
        selector_ = SelectKBest(score_func=f_classif, k="all")
        selector_.fit(X, y)
        pvals = selector_.pvalues_
        pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)
        scores = selector_.scores_
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        num_sign_features = np.sum((pvals < 0.05) & (scores > 5))
        min_features = max(MIN_FEATURES, int(num_sign_features / 2))
        max_features = int(num_sign_features * 2)
        if max_features > X.shape[1]:
            max_features = X.shape[1]
        if max_features > MAX_FEATURES:
            max_features = MAX_FEATURES
        max_features = max(max_features, int(X.shape[1] / 2))
        min_features = int(min_features)
        max_features = int(max_features)
        if (num_sign_features > min_features) and (num_sign_features < max_features):
            seed_features = int(num_sign_features)
        else:
            seed_features = int((min_features + max_features) / 2)
        logger.info(f"Initial limits for k_features: {min_features} - {max_features}, seed: {seed_features}")
        return min_features, max_features, seed_features

    min_k_features, max_k_features, seed_k_features = dense_k_features_initial_limits(X, y)

    def objective(trial):
        k_features = trial.suggest_int("k_features", min_k_features, max_k_features, step=1)
        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
        reducer = KFeaturesReducer(k_features=k_features)
        clf = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=5,
            random_state=42
        )
        pipe = Pipeline([
            ("reduce", reducer),
            ("clf", clf)
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]
            score = roc_auc_score(y_te, proba)
            scores.append(score)
            trial.report(np.mean(scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    initial_params = {
        "k_features": seed_k_features,
        "alpha": 1e-4,
    }
    study.enqueue_trial(params=initial_params)
    study.optimize(objective, n_trials=NUM_TRIALS, show_progress_bar=True)
    logger.info("Best trial:")
    logger.info(f"  ROC-AUC: {study.best_value}")
    logger.info(f"  Params: {study.best_params}")
    return study.best_params


def get_latent_space_parameters(X, y, k_features):

    def sparse_n_components_initial_limits(X, y, k_features):
        kfeat = KFeaturesReducer(k_features=k_features)
        kfeat.fit(X, y)
        X_red = kfeat.transform(X)
        max_n_components = X_red.shape[0] - 1
        min_n_components = MIN_FEATURES
        min_n_components = int(min_n_components)
        max_n_components = int(max_n_components)
        max_n_components = min(X_red.shape[1]-1, max_n_components)
        reducer_ = PCA(n_components=max_n_components, svd_solver="arpack")
        reducer_.fit(X_red)
        ideal_n_components = reducer_.n_components_
        if (ideal_n_components > min_n_components) and (ideal_n_components < max_n_components):
            seed_n_components = int(ideal_n_components)
        else:
            seed_n_components = int((min_n_components + max_n_components) / 2)
        logger.info(f"Initial limits for n_components: {min_n_components} - {max_n_components}, seed: {seed_n_components}")
        return min_n_components, max_n_components, seed_n_components

    logger.info("Estimating initial limits for n_components...")
    min_n_components, max_n_components, seed_n_components = sparse_n_components_initial_limits(X, y, k_features)
    logger.info(f"Estimated limits for n_components: {min_n_components} - {max_n_components}, seed: {seed_n_components}")

    logger.info("Preparing cross-validation folds to identify a reasonable latent dimensionality...")
    folds_full_latent = []
    splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.20, random_state=42)
    for train_idx, test_idx in splitter.split(X, y):
        logger.info(f"  Fold with {len(train_idx)} training and {len(test_idx)} test samples")
        kfeat = KFeaturesReducer(k_features=k_features)
        kfeat.fit(X[train_idx], y[train_idx])
        X_tr = kfeat.transform(X[train_idx])
        X_te = kfeat.transform(X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]
        reducer = PCA(max_n_components, svd_solver="arpack", random_state=42)
        X_tr = reducer.fit_transform(X_tr)
        X_te = reducer.transform(X_te)
        folds_full_latent += [(X_tr, y_tr, X_te, y_te)]

    def objective(trial):
        n_components = trial.suggest_int("n_components", min_n_components, max_n_components, step=1)
        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
        clf = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=5,
            random_state=42
        )
        scores = []
        for fold_idx, fold in enumerate(folds_full_latent):
            X_tr, y_tr, X_te, y_te = fold
            X_tr_reduced = X_tr[:, :n_components]
            X_te_reduced = X_te[:, :n_components]
            clf.fit(X_tr_reduced, y_tr)
            proba = clf.predict_proba(X_te_reduced)[:, 1]
            score = roc_auc_score(y_te, proba)
            scores.append(score)
            trial.report(np.mean(scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))
    
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    initial_params = {
        "n_components": seed_n_components,
        "alpha": 1e-4,
    }
    study.enqueue_trial(params=initial_params)
    study.optimize(objective, n_trials=NUM_TRIALS, show_progress_bar=True)
    logger.info("Best trial:")
    logger.info(f"  ROC-AUC: {study.best_value}")
    logger.info(f"  Params: {study.best_params}")
    return study.best_params


def get_reducer_parameters(X, y):
    k_features_params = get_k_features_parameter(X, y)
    n_components_params = get_latent_space_parameters(X, y, k_features_params["k_features"])
    return {
        "k_features": k_features_params["k_features"],
        "n_components": n_components_params["n_components"],
    }


class DenseDimReducerBinaryClassification(object):
    def __init__(self, 
                 n_components, 
                 k_features,
                 random_state=None):
        self.n_components = n_components
        self.k_features = k_features
        self.random_state = random_state

    def fit(self, X, y):
        self.var_filter_ = VarianceThreshold(threshold=0.0)
        X_nonconst = self.var_filter_.fit_transform(X)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_nonconst)
        k = min(self.k_features, X.shape[1])
        self.selector_ = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.selector_.fit_transform(X_scaled, y)
        n_comp = min(self.n_components, X_selected.shape[1], X_selected.shape[0]-1)
        self.reducer_ = PCA(
            n_components=n_comp,
            random_state=self.random_state,
            svd_solver="auto"
        )
        self.reducer_.fit(X_selected)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["var_filter_", "scaler_", "selector_", "reducer_"])
        X_nonconst = self.var_filter_.transform(X)
        X_scaled = self.scaler_.transform(X_nonconst)
        X_selected = self.selector_.transform(X_scaled)
        return self.reducer_.transform(X_selected)