import json
import os

import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import onnx

from ..utils.logging import logger


MIN_FEATURES = 4
MAX_FEATURES = 1024

NUM_TRIALS = 1 # TODO increase to 20 or 50 later


def decide_if_latent_variables(X, y):
    """
    Determines whether latent variable generation should be performed based on the 
    characteristics of the input data and the performance of a classifier with and 
    without dimensionality reduction.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Target vector of shape (n_samples,).
    
    Returns
    -------
    bool
        True if latent variable generation should be performed, False otherwise.
    
    Notes
    -----
    - If the number of features is less than or equal to `MIN_FEATURES`, latent variable 
      generation is not performed.
    - If the number of features is greater than `MAX_FEATURES` or greater than the number 
      of samples, latent variable generation is performed.
    - Otherwise, the function evaluates the performance of a classifier with and without 
      dimensionality reduction using PCA and compares their mean ROC AUC scores. If the 
      classifier with dimensionality reduction performs better, latent variable generation 
      is performed.
    
    Logging
    -------
    - Logs the decision-making process and the mean ROC AUC scores for both cases.
    """

    logger.info("Deciding whether to perform latent variable generation...")

    if X.shape[1] <= MIN_FEATURES:
        logger.info("Number of features is less than or equal to {0}. No latent variable generation.".format(MIN_FEATURES))
        return False
    
    if X.shape[1] > MAX_FEATURES:
        logger.info("Number of features is greater than {0}. Latent variable generation will be performed.".format(MAX_FEATURES))
        return True
    
    if X.shape[1] > X.shape[0]:
        logger.info("Number of features is greater than number of samples. Latent variable generation will be performed.")
        return True
    
    splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

    scores_reduced = []
    scores_vanilla = []    
    for train_index, test_index in splitter.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        n_components = min(X_train.shape[1], X_train.shape[0]) - 1
        n_components = min(n_components, MAX_FEATURES)
        reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        reducer.fit(X_train)
        explained_variance_ratio_cumsum = np.cumsum(reducer.explained_variance_ratio_)
        logger.info(f"Explained variance ratio cumulative sum: {explained_variance_ratio_cumsum}")
        n_components_90 = np.searchsorted(explained_variance_ratio_cumsum, 0.9) + 1
        logger.info(f"Number of components explaining 90% of variance: {n_components_90}")
        X_train_reduced = reducer.transform(X_train)[:, :n_components_90]
        X_test_reduced = reducer.transform(X_test)[:, :n_components_90]
        clf_reduced = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=5,
            random_state=42
        )
        clf_vanilla = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=5,
            random_state=42
        )
        clf_reduced.fit(X_train_reduced, y_train)
        clf_vanilla.fit(X_train, y_train)

        y_pred_reduced = clf_reduced.predict_proba(X_test_reduced)[:, 1]
        y_pred_vanilla = clf_vanilla.predict_proba(X_test)[:, 1]
        scores_reduced += [roc_auc_score(y_test, y_pred_reduced)]
        scores_vanilla += [roc_auc_score(y_test, y_pred_vanilla)]

    scores_reduced = np.array(scores_reduced)
    scores_vanilla = np.array(scores_vanilla)

    if not (np.mean(scores_vanilla) - np.mean(scores_reduced)) > 0.2:
        logger.info("Latent variable generation will be performed. Mean ROC AUC with latent variables: {0:.3f}, without latent variables: {1:.3f}".format(np.mean(scores_reduced), np.mean(scores_vanilla)))
        return True
    else:
        logger.info("No latent variable generation. Mean ROC AUC with latent variables: {0:.3f}, without latent variables: {1:.3f}".format(np.mean(scores_reduced), np.mean(scores_vanilla)))
    return False


def find_latent_params(X, y):
    """
    Optimize the number of latent components and regularization parameter for binary classification 
    using Principal Component Analysis (PCA) and Stochastic Gradient Descent Classifier (SGDClassifier).
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    y : array-like of shape (n_samples,)
        The target binary labels.
    
    Returns
    -------
    results : dict
        A dictionary containing the optimal number of latent components:
        - "n_components": int, the optimal number of components.
    
    Notes
    -----
    - The function uses PCA to reduce dimensionality and Optuna for hyperparameter optimization.
    - The optimization process includes pruning of unpromising trials using a MedianPruner.
    - The ROC-AUC score is used as the evaluation metric for model performance.
    - The function performs stratified shuffle split cross-validation to ensure balanced class distribution 
      in training and testing sets.
    """
    do_latent = False
    #do_latent = decide_if_latent_variables(X, y) # TODO uncomment
    if not do_latent:
        logger.info("Skipping latent variable generation.")
        return {
            "n_components": None
        }
    logger.info("Finding optimal latent variable parameters...")

    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

    min_n_components = []
    max_n_components = []
    seed_n_components = []

    logger.debug("Preparing folds for cross-validation...")
    folds = []
    for train_index, test_index in cv.split(X, y):
        logger.debug("Precomputing reductions for a fold...")
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        logger.debug(f"Train shape: {X_tr.shape}, Test shape: {X_te.shape}")
        n_components = min(X_tr.shape[1], X_tr.shape[0]) - 1
        n_components = min(n_components, MAX_FEATURES)
        logger.debug(f"Using n_components={n_components} for PCA.")
        reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        reducer.fit(X_tr)
        X_tr = reducer.transform(X_tr)
        X_te = reducer.transform(X_te)
        folds += [(X_tr, X_te, y_tr, y_te)]
        explained_variance_ratio_cumsum = np.cumsum(reducer.explained_variance_ratio_)
        n_components_80 = np.searchsorted(explained_variance_ratio_cumsum, 0.8) + 1
        n_components_90 = np.searchsorted(explained_variance_ratio_cumsum, 0.9) + 1
        n_components_99 = np.searchsorted(explained_variance_ratio_cumsum, 0.9) + 1
        min_n_components += [n_components_80]
        seed_n_components += [n_components_90]
        max_n_components += [n_components_99]
    
    min_n_components = int(np.mean(min_n_components))
    seed_n_components = int(np.mean(seed_n_components))
    max_n_components = int(np.mean(max_n_components))
        
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
        for fold_idx, (X_tr, X_te, y_tr, y_te) in enumerate(folds):
            X_tr = X_tr[:, :n_components]
            X_te = X_te[:, :n_components]
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
        "n_components": seed_n_components,
        "alpha": 1e-4,
    }
    study.enqueue_trial(params=initial_params)
    study.optimize(objective, n_trials=NUM_TRIALS, show_progress_bar=True)
    logger.info("Best trial:")
    logger.info(f"  ROC-AUC: {study.best_value}")
    logger.info(f"  Params: {study.best_params}")
    
    results = {
        "n_components": study.best_params["n_components"]
    }

    return results


class LatentVariablesForBinaryClassification(object):
    """
    A class for reducing the dimensionality of data for binary classification tasks 
    using Principal Component Analysis (PCA).
    
    Parameters
    ----------
    n_components : int
        The number of principal components to retain during dimensionality reduction.
    
    Methods
    -------
    fit(X, y=None)
        Fits the PCA reducer to the input data.
    transform(X, y=None)
        Transforms the input data using the fitted PCA reducer.
    save(model_dir: str)
        Saves the PCA reducer and its metadata to the specified directory.
    load(model_dir: str)
        Loads the PCA reducer and its metadata from the specified directory.
    """

    def __init__(self, n_components: int=None):
        """
        Initializes the class with the specified number of components.
        Parameters
        ----------
        n_components : int
            The number of components to use for the binary classification model.
        """
        self.n_components = n_components

    def fit(self, X, y=None):
        """
        Fit the latent variable reducer using the provided data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit the latent variable reducer.
        y : array-like of shape (n_samples,), optional
            The target values (ignored in this method, included for compatibility).
        
        Returns
        -------
        self : object
            Returns the instance of the latent variable reducer.
        """
        if self.n_components is None:
            self.reducer = None
            return self
        logger.info("Fitting latent reducer with {0} components...".format(self.n_components))
        n_components = min(self.n_components, X.shape[1])
        self.reducer = PCA(n_components=n_components, random_state=42)
        self.reducer.fit(X)
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input data using the fitted dimensionality reducer.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to transform.
        y : None, optional
            Ignored. This parameter exists for compatibility with scikit-learn
            transformers.
        
        Returns
        -------
        X : array-like of shape (n_samples, n_components)
            The transformed data.
        
        Raises
        ------
        RuntimeError
            If the reducer has not been fitted prior to calling this method.
        """
        if not hasattr(self, "reducer"):
            raise RuntimeError("The reducer has not been fitted yet. Please call 'fit' before 'transform'.")
        if self.reducer is None:
            return X
        logger.info("Transforming latent reducer using PCA...")
        X = self.reducer.transform(X)
        return X
    
    def save(self, model_dir: str):
        """
        Save the latent variable reducer to the specified directory.

        This method saves the metadata and the reducer object to the given directory.
        If the directory does not exist, it will be created.

        Parameters
        ----------
        model_dir : str
            The directory where the latent variable reducer and its metadata will be saved.

        
        Notes
        -----
        - The metadata is saved as a JSON file named `latent_reducer_metadata.json`.
        - The reducer object is serialized and saved as a joblib file named `latent_reducer.joblib`.
        """
        if not os.path.exists(model_dir):
            logger.info(f"Creating directory {model_dir} for saving the latent reducer.")
            os.makedirs(model_dir)
        metadata = {
            "n_components": self.n_components,
        }
        meta_path = os.path.join(model_dir, "latent_reducer_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        reducer_path = os.path.join(model_dir, "latent_reducer.joblib")
        joblib.dump(self.reducer, reducer_path)

    @classmethod
    def load(cls, model_dir: str):
        """
        Load a latent variable reducer object from a specified directory.

        Parameters
        ----------
        model_dir : str
            The directory where the latent variable reducer model and metadata are stored.

        Returns
        -------
        cls
            An instance of the class with the loaded reducer and metadata.

        Raises
        ------
        FileNotFoundError
            If the specified directory does not exist.
            If the metadata file "latent_reducer_metadata.json" is not found in the directory.
            If the reducer file "latent_reducer.joblib" is not found in the directory.

        Notes
        -----
        The method expects the directory to contain two files:
        1. "latent_reducer_metadata.json" - A JSON file containing metadata such as the number of components.
        2. "latent_reducer.joblib" - A serialized file containing the reducer object.
        """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"The directory {model_dir} does not exist.")
        meta_path = os.path.join(model_dir, "latent_reducer_metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"The metadata file {meta_path} does not exist in the directory {model_dir}.")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        obj = cls(n_components=metadata["n_components"])
        reducer_path = os.path.join(model_dir, "latent_reducer.joblib")
        if not os.path.exists(reducer_path):
            raise FileNotFoundError(f"The reducer file {reducer_path} does not exist in the directory {model_dir}.")
        obj.reducer = joblib.load(reducer_path)
        return obj


class PCALayer(nn.Module):
    def __init__(self, components, mean):
        super().__init__()
        self.register_buffer("components", torch.tensor(components, dtype=torch.float32))
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
    
    def forward(self, x):
        x = x - self.mean
        return torch.matmul(x, self.components.T)


from .. import ONNX_TARGET_OPSET, ONNX_IR_VERSION

def convert_to_onnx(model_dir: str):
    """
    Converts a latent variable reducer model to ONNX format and saves it to the specified directory.
    This function loads a latent variable reducer model from the given directory, checks if the reducer exists, 
    and converts it to an ONNX model. The ONNX model is then saved as "latent_reducer.onnx" in the same directory.
    
    Parameters
    ----------
    model_dir : str
        The directory where the latent variable reducer model is stored and where the ONNX model will be saved.
    
    Returns
    -------
    None
        This function does not return any value. It saves the ONNX model to the specified directory.
    
    Notes
    -----
    - If the latent variable reducer does not exist, the function logs a message and exits without performing any conversion.
    - The ONNX model is created using PyTorch's `torch.onnx.export` function.
    - The input and output tensors of the ONNX model are configured to support dynamic batch sizes.
    
    Examples
    --------
    >>> convert_to_onnx("/path/to/model_directory")
    """
    latent_reducer = LatentVariablesForBinaryClassification.load(model_dir)
    if latent_reducer.reducer is None:
        logger.info("No latent reducer to convert to ONNX.")
        return None
    reducer = latent_reducer.reducer
    logger.info("Converting latent reducer to ONNX via PyTorch")
    pca_layer = PCALayer(reducer.components_, reducer.mean_)
    dummy_input = torch.randn(1, reducer.mean_.shape[0], dtype=torch.float32)
    onnx_path = os.path.join(model_dir, "latent_reducer.onnx")
    torch.onnx.export(
        pca_layer,
        dummy_input,
        onnx_path,
        input_names=["input_latent"],
        output_names=["output_latent"],
        dynamic_axes={"input_latent": {0: "batch_size"}, "output_latent": {0: "batch_size"}},
        opset_version=ONNX_TARGET_OPSET,
    )
    
    model = onnx.load(onnx_path)
    model.graph.name = "LatentReducer"
    model.ir_version = ONNX_IR_VERSION

    for node in model.graph.node:
        if node.name:
            node.name = f"{node.name}_latent"

    onnx.save(model, onnx_path)
    logger.info(f"Set ONNX IR version to {ONNX_IR_VERSION} for {onnx_path}")
    logger.info(f"Latent reducer ONNX model saved to {onnx_path}")
    return onnx_path