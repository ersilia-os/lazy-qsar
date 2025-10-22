import os
import numpy as np
import optuna
import json
import joblib
import warnings
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

from ...utils.logging import logger
from ... import ONNX_TARGET_OPSET, ONNX_IR_VERSION

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


MIN_FEATURES = 4
MAX_FEATURES = 2048

MAX_NUM_TRIALS = 10


def find_params(X, y, num_trials):
    """
    Perform feature selection and hyperparameter optimization for binary classification.
    This function uses `SelectKBest` to determine the most relevant features based on
    ANOVA F-values and employs `optuna` for hyperparameter optimization of the number
    of features (`k_features`) and the regularization strength (`alpha`) of an
    `SGDClassifier`. The optimization aims to maximize the ROC-AUC score.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Target vector of shape (n_samples,).

    Returns
    -------
    dict
        A dictionary containing the optimal number of features:
        - "k_features" : int
            The optimal number of features selected.

    Notes
    -----
    - The function splits the data into stratified folds for cross-validation.
    - The `optuna` library is used for hyperparameter optimization with a pruning mechanism.
    - The initial limits for `k_features` are determined based on the significance
      of features (p-values and scores).
    - The `SGDClassifier` is used with logistic loss and early stopping enabled.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> result = find_feature_selection_parameters(X, y)
    >>> print(result)
    {'k_features': 10}
    """

    do_feature_selection = True
    if not do_feature_selection:
        return {"k_features": None}

    def k_features_initial_limits(X, y):
        if X.shape[1] < MIN_FEATURES:
            return None, None
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
        logger.info(
            f"Initial limits for k_features: {min_features} - {max_features}, seed: {seed_features}"
        )
        return min_features, max_features, seed_features

    folds = []
    splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    for train_idx, test_idx in splitter.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        selector = SelectKBest(score_func=f_classif, k="all")
        selector.fit(X_tr, y_tr)
        scores = selector.scores_
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        features_idxs = np.argsort(scores)[::-1]
        X_tr = X_tr[:, features_idxs]
        X_te = X_te[:, features_idxs]
        folds += [(X_tr, X_te, y_tr, y_te)]

    min_k_features, max_k_features, seed_k_features = k_features_initial_limits(X, y)

    def objective(trial):
        k_features = trial.suggest_int(
            "k_features", min_k_features, max_k_features, step=1
        )
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
            random_state=42,
        )
        scores = []
        for fold_idx, (X_tr, X_te, y_tr, y_te) in enumerate(folds):
            X_tr = X_tr[:, :k_features]
            X_te = X_te[:, :k_features]
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)[:, 1]
            score = roc_auc_score(y_te, proba)
            scores += [score]
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
    study.optimize(
        objective, n_trials=min(num_trials, MAX_NUM_TRIALS), show_progress_bar=True
    )
    logger.info("Best trial:")
    logger.info(f"  ROC-AUC: {study.best_value}")
    logger.info(f"  Params: {study.best_params}")

    results = {"k_features": study.best_params["k_features"]}

    return results


class FeatureSelector(object):
    """
    A feature selector for binary classification tasks using SelectKBest.

    This class provides functionality to select the top `k` features based on
    statistical tests for binary classification problems. It supports saving
    and loading the feature selector for reuse.

    Parameters
    ----------
    k_features : int, optional
        The number of top features to select. If None, all features are selected.

    Methods
    -------
    fit(X)
        Fits the feature selector to the data.
    transform(X)
        Transforms the data by selecting the top `k` features.
    save(model_dir)
        Saves the feature selector and its metadata to the specified directory.
    load(model_dir)
        Loads the feature selector and its metadata from the specified directory.

    Raises
    ------
    ValueError
        If the feature selector is not fitted before calling `transform`.
        If the specified model directory or required files do not exist during `load`.
    """

    def __init__(self, k_features: int = None):
        """
        Initializes the FeatureSelectorForBinaryClassification.

        Parameters
        ----------
        k_features : int, optional
            The number of top features to select. If None, all features are selected.
        """
        self.k_features = k_features

    def fit(self, X, y):
        """
        Fits the SelectKBest feature selector to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit the feature selector on.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.k_features is None:
            self.selector = None
            return self
        k_features = (
            min(self.k_features, X.shape[1]) if self.k_features is not None else "all"
        )
        self.selector = SelectKBest(score_func=f_classif, k=k_features)
        self.selector.fit(X, y)
        return self

    def transform(self, X, y=None):
        """
        Transform the input data using the fitted feature selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_selected_features)
            The transformed data with selected features.

        Raises
        ------
        ValueError
            If the feature selector has not been fitted yet.
        """
        if not hasattr(self, "selector"):
            raise ValueError("The feature selector has not been fitted yet.")
        if self.selector is None:
            return X
        X = self.selector.transform(X)
        return X

    def save(self, name: str, model_dir: str):
        """
        Save the feature selector and its metadata to the specified directory.
        This method saves the metadata (e.g., number of selected features) as a JSON file
        and the feature selector object as a joblib file in the given directory.

        Parameters
        ----------
        model_dir : str
            The directory where the metadata and feature selector will be saved.

        Raises
        ------
        OSError
            If there is an issue creating or writing to the specified directory.
        """
        metadata = {
            "k_features": self.k_features,
        }
        meta_path = os.path.join(model_dir, f"{name}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        joblib_path = os.path.join(model_dir, f"{name}.joblib")
        joblib.dump(self.selector, joblib_path)

    @classmethod
    def load(cls, name: str, model_dir: str):
        """
        Load a feature selector object from a specified directory.

        Parameters
        ----------
        model_dir : str
            Path to the directory containing the saved feature selector files.

        Returns
        -------
        cls
            An instance of the class with the loaded feature selector.

        Raises
        ------
        ValueError
            If the specified model directory does not exist.
            If the metadata file `feature_selector_metadata.json` does not exist.
            If the selector file `feature_selector.joblib` does not exist.

        Notes
        -----
        The method expects the following files to be present in the `model_dir`:
        - `feature_selector_metadata.json`: Contains metadata about the feature selector.
        - `feature_selector.joblib`: Serialized feature selector object.
        """
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist.")
        meta_path = os.path.join(model_dir, f"{name}_metadata.json")
        if not os.path.exists(meta_path):
            raise ValueError(f"Metadata file {meta_path} does not exist.")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        k_features = metadata.get("k_features", None)
        selector_path = os.path.join(model_dir, f"{name}.joblib")
        if not os.path.exists(selector_path):
            raise ValueError(f"Selector file {selector_path} does not exist.")
        selector = joblib.load(selector_path)
        obj = cls(k_features=k_features)
        obj.selector = selector
        return obj


def convert_to_onnx(name, model_dir: str):
    """
    Converts a feature selector model to ONNX format and saves it to the specified directory.

    Parameters
    ----------
    model_dir : str
        The directory where the feature selector model is stored and where the ONNX file will be saved.

    Notes
    -----
    - If no feature selection was performed (i.e., `feature_selector.selector` is None), the function logs a message
      and skips the ONNX conversion.
    - The ONNX model is saved as "feature_selector.onnx" in the specified `model_dir`.

    Raises
    ------
    FileNotFoundError
        If the specified `model_dir` does not exist.
    ValueError
        If the `selector` object is not compatible with ONNX conversion.
    """

    feature_selector = FeatureSelector.load(name, model_dir)
    if feature_selector.selector is None:
        logger.info("No feature selection was performed. Skipping ONNX conversion.")
        return None

    selector = feature_selector.selector
    initial_type = [
        (f"input_{name}", FloatTensorType([None, selector.scores_.shape[0]]))
    ]
    onnx_model = skl2onnx.convert_sklearn(
        selector, initial_types=initial_type, target_opset=ONNX_TARGET_OPSET
    )

    onnx_model.graph.name = f"{name}"
    onnx_model.ir_version = ONNX_IR_VERSION
    onnx_model.graph.input[0].name = f"input_{name}"
    onnx_model.graph.output[0].name = f"output_{name}"

    for node in onnx_model.graph.node:
        if f"_{name}" not in node.name:
            node.name = f"{node.name}_{name}"

    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info(f"Feature selector converted to ONNX and saved at {onnx_path}.")
    return onnx_path
