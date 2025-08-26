import os
import json
import numpy as np

from .descriptors import ChemeleonDescriptor, MorganFingerprint
from .models import (
    LazyRandomForestBinaryClassifier,
    LazyTuneTablesBinaryClassifier,
    LazyLogisticRegressionBinaryClassifier,
)

from .config.presets import preset_params
from .utils.logging import logger

descriptors_dict = {"chemeleon": ChemeleonDescriptor, "morgan": MorganFingerprint}


models_dict = {
    "tune_tables": LazyTuneTablesBinaryClassifier,
    "random_forest": LazyRandomForestBinaryClassifier,
    "logistic_regression": LazyLogisticRegressionBinaryClassifier,
}

models_dict = dict((k, v) for k, v in models_dict.items() if v is not None)


class LazyBinaryQSAR(object):

    def __init__(
        self, descriptor_type: str = "chemeleon", model_type: str = "random_forest", mode: str = "default", **kwargs
    ):
        """
        Initialize a LazyBinaryQSAR

        This class serves as a wrapper for various binary classification models for chemistry,
        allowing for easy switching between different algorithms and configurations.
        Args:
            descriptor_type (str): The type of descriptor to use. Options are 'chemeleon' (default) or 'morgan'.
            model_type (str): The type of model to use. Options are 'random_forest' (default), 'logistic_regression' or 'tune_tables'.
            mode (str): The preset mode to use for the model parameters. Options are 'quick' or 'default'.
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Usage:
        >>> from lazyqsar.qsar import LazyBinaryQSAR
        >>> qsar = LazyBinaryQSAR(descriptor_type="chemeleon", model_type="random_forest", mode="default")
        >>> qsar.fit(smiles_list, y)
        >>> predictions = qsar.predict_proba(smiles_list)
        """
        self.descriptor_type = descriptor_type
        self.model_type = model_type
        self.mode = mode

        if mode not in preset_params:
            raise ValueError(f"Unsupported mode: {mode}. Please choose from {list(preset_params.keys())}.")
        
        presets = preset_params[mode]
        combined_kwargs = {**presets, **kwargs}

        if descriptor_type not in descriptors_dict:
            raise ValueError(f"Unsupported descriptor type: {descriptor_type}")
        self.descriptor = descriptors_dict[descriptor_type]()

        if model_type not in models_dict:
            raise ValueError(f"Unsupported model type: {model_type}")
        else:
            self.model = models_dict[model_type](**combined_kwargs)

    def fit(self, smiles_list, y):
        y = np.array(y, dtype=int)
        logger.debug(f"Fitting inputs to feature descriptors using {self.descriptor_type}")
        self.descriptor.fit(smiles_list)
        logger.debug(
            f"Transforming inputs to feature descriptors using {self.descriptor_type}"
        )
        descriptors = self.descriptor.transform(smiles_list)
        logger.debug(f"Performing predictions on input feature of shape: {descriptors.shape}")
        self.model.fit(X=descriptors, y=y)

    def predict_proba(self, smiles_list):
        logger.debug(
            f"Transforming inputs to feature descriptors using {self.descriptor_type}"
        )
        descriptors = self.descriptor.transform(smiles_list)
        logger.array(f"Performing predictions on input feature of shape: {descriptors.shape}")
        y_hat_1 = np.array(self.model.predict(descriptors))
        y_hat_0 = 1 - y_hat_1
        return np.array([y_hat_0, y_hat_1]).T
    
    def predict(self, smiles_list, threshold=0.5):
        y_hat = self.predict_proba(smiles_list)[:, 1]
        y_bin = []
        for y in y_hat:
            if y >= threshold:
                y_bin.append(1)
            else:
                y_bin.append(0)
        return np.array(y_bin, dtype=int)

    def save_model(self, model_dir: str):
        logger.debug(f"LazyQSAR Saving model to {model_dir}")
        config = {
            "descriptor_type": self.descriptor_type,
            "model_type": self.model_type,
        }
        self.model.save_model(model_dir)
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        metadata["descriptor_type"] = self.descriptor_type
        metadata["model_type"] = self.model_type
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)
        self.descriptor.save(model_dir)
        logger.info(f"The model is successfully saved at {model_dir}")

    @classmethod
    def load_model(cls, model_dir: str):
        logger.debug(f"LazyQSAR Loading model from {model_dir}")
        obj = cls()
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        descriptor_type = config["descriptor_type"]
        model_type = config["model_type"]
        obj.descriptor_type = descriptor_type
        obj.model_type = model_type
        obj.descriptor = descriptors_dict[descriptor_type].load(model_dir)
        obj.model = models_dict[model_type].load_model(model_dir)
        logger.info(f"Model successfully loaded from {model_dir}")
        return obj
