import json
import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from .. import ONNX_TARGET_OPSET, ONNX_IR_VERSION

from ..utils.logging import logger


def find_params(X):
    """
    Checks if the input data is sparse.

    Parameters
    ----------
    X : ndarray
        Input data.

    Returns
    -------
    bool
        True if the data is sparse, False otherwise.
    """
    tot = X.shape[0] * X.shape[1]
    n_zero = np.sum(X == 0)
    sparsity = n_zero / tot
    is_sparse = sparsity > 0.9

    results = {
        "is_sparse": is_sparse,
    }

    return results


class Preprocessor(object):
    """
    A class for preprocessing data, including handling sparsity,
    removing constant features, imputing missing values, and scaling.

    Attributes
    ----------
    is_sparse : bool
        Indicates whether the input data is sparse.
    var_thr : VarianceThreshold
        VarianceThreshold object for removing constant features.
    imputer : SimpleImputer
        SimpleImputer object for imputing missing values.
    scaler : TfidfTransformer or StandardScaler
        Transformer or scaler for normalizing the data.
    """

    def __init__(self, is_sparse: bool):
        """
        Initializes the Preprocessor object.
        """
        self.is_sparse = is_sparse

    def fit(self, X):
        """
        Fits the preprocessor to the input data.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        Preprocessor
            The fitted Preprocessor object.
        """
        logger.info("Fitting the preprocessor...")
        logger.info("Shape of the input data: {0}".format(X.shape))
        logger.info("Removing constant features")
        self.input_dim = X.shape[1]
        logger.info("Fitting a simple median imputer")
        self.imputer = SimpleImputer(strategy="median")
        self.imputer.fit(X)
        X = self.imputer.transform(X)
        self.var_thr = VarianceThreshold(threshold=0.0)
        self.var_thr.fit(X)
        X = self.var_thr.transform(X)
        if self.is_sparse:
            logger.info("Data is sparse, TF-IDF preprocessor will be used")
            self.scaler = TfidfTransformer()
            self.scaler.fit(X)
        else:
            logger.info("Data is dense, scaler will be used")
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        return self

    def transform(self, X):
        """
        Transforms the input data using the fitted preprocessor.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray
            Transformed data.

        Raises
        ------
        ValueError
            If the preprocessor has not been fitted.
        """
        if not hasattr(self, "var_thr") or self.var_thr is None:
            raise ValueError("Preprocessor not fitted. Call `fit` first.")
        logger.info("Transforming the data using the fitted preprocessor...")
        X = self.imputer.transform(X)
        X = self.var_thr.transform(X)
        X = (
            self.scaler.transform(X).toarray()
            if self.is_sparse
            else self.scaler.transform(X)
        )
        return X

    def save(self, name: str, model_dir: str):
        """
        Saves the fitted preprocessor to the specified directory.

        Parameters
        ----------
        model_dir : str
            Directory where the preprocessor will be saved.

        Raises
        ------
        ValueError
            If the preprocessor has not been fitted.
        """
        if not hasattr(self, "var_thr") or self.var_thr is None:
            raise ValueError("Preprocessor not fitted. Call `fit` first.")

        if not os.path.exists(model_dir):
            logger.info(f"Creating directory {model_dir} for saving the preprocessor.")
            os.makedirs(model_dir)

        metadata = {
            "is_sparse": bool(self.is_sparse),
            "input_dim": int(self.input_dim),
        }
        meta_path = os.path.join(model_dir, f"{name}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        imputer_path = os.path.join(model_dir, f"{name}_imputer.joblib")
        joblib.dump(self.imputer, imputer_path)
        var_thr_path = os.path.join(model_dir, f"{name}_var_thr.joblib")
        joblib.dump(self.var_thr, var_thr_path)
        scaler_path = os.path.join(model_dir, f"{name}_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load(cls, name: str, model_dir: str):
        """
        Loads a preprocessor from the specified directory.

        Parameters
        ----------
        model_dir : str
            Directory from which the preprocessor will be loaded.

        Returns
        -------
        Preprocessor
            The loaded Preprocessor object.

        Raises
        ------
        FileNotFoundError
            If the specified directory or required files do not exist.
        """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
        meta_path = os.path.join(model_dir, f"{name}_metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file {meta_path} not found.")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        obj = cls(is_sparse=bool(metadata["is_sparse"]))
        obj.input_dim = metadata["input_dim"]
        imputer_path = os.path.join(model_dir, f"{name}_imputer.joblib")
        if not os.path.exists(imputer_path):
            raise FileNotFoundError(f"Imputer file {imputer_path} not found.")
        obj.imputer = joblib.load(imputer_path)
        var_thr_path = os.path.join(model_dir, f"{name}_var_thr.joblib")
        if not os.path.exists(var_thr_path):
            raise FileNotFoundError(f"VarianceThreshold file {var_thr_path} not found.")
        obj.var_thr = joblib.load(var_thr_path)
        scaler_path = os.path.join(model_dir, f"{name}_scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file {scaler_path} not found.")
        obj.scaler = joblib.load(scaler_path)
        return obj


def convert_to_onnx(name: str, model_dir: str):
    """
    Convert to ONNX format and save the preprocessor.
    It creates a file named preprocessor.onnx in the specified directory.

    Parameters
    ----------
    model_dir : str
        Directory where the preprocessor is saved.

    Returns
    -------
    str
        Path to the saved ONNX file.

    Raises
    ------
    FileNotFoundError
        If the specified directory or required files do not exist.
    ValueError
        If the preprocessor has not been fitted.
    """
    preprocessor = Preprocessor.load(name, model_dir)
    pipe = Pipeline(
        [
            ("imputer_preprocessor", preprocessor.imputer),
            ("varthr_preprocessor", preprocessor.var_thr),
            ("scaler_preprocessor", preprocessor.scaler),
        ]
    )
    initial_type = [(f"input_{name}", FloatTensorType([None, preprocessor.input_dim]))]
    onnx_model = convert_sklearn(
        pipe,
        initial_types=initial_type,
        target_opset={"": ONNX_TARGET_OPSET, "ai.onnx.ml": ONNX_TARGET_OPSET},
    )

    onnx_model.ir_version = ONNX_IR_VERSION
    onnx_model.graph.name = f"{name}"
    onnx_model.graph.output[0].name = f"output_{name}"
    onnx_model.graph.input[0].name = f"input_{name}"

    for i, node in enumerate(onnx_model.graph.node):
        if name not in node.name:
            if node.name:
                node.name = node.name + "_" + name
            else:
                node.name = "node_{0}_{1}".format(i, name)

    onnx_path = os.path.join(model_dir, f"{name}.onnx")
    logger.info(f"Saving the preprocessor ONNX model to {onnx_path}")

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.debug("ONNX preprocessor model properties:")
    logger.debug(f"Input name: {onnx_model.graph.input[0].name}")
    logger.debug(f"Input shape: {onnx_model.graph.input[0].type.tensor_type.shape}")
    logger.debug(f"Output name: {onnx_model.graph.output[0].name}")
    logger.debug(f"Output shape: {onnx_model.graph.output[0].type.tensor_type.shape}")

    return onnx_path
