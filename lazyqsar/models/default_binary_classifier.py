import json
import multiprocessing
import os
import random
import shutil
import time

import h5py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

from ..feature_selection.feature_selection_for_binary_classification import (
    FeatureSelectorForBinaryClassification,
    find_feature_selector_params,
)
from ..heads.head_for_binary_classification import (
    HeadForBinaryClassification,
    find_head_params,
)
from ..latent_variables.latent_variables_for_binary_classification import (
    LatentVariablesForBinaryClassification,
    find_latent_params,
)
from ..preprocess.preprocess import Preprocessor, find_preprocessor_params
from ..utils.deciders import BinaryClassifierMaxSamplesDecider
from ..utils.io import InputUtils
from ..utils.samplers import BinaryClassifierSamplingUtils as SamplingUtils

from ..utils.logging import logger

from ..preprocess.preprocess import convert_to_onnx as convert_preprocess_to_onnx
from ..feature_selection.feature_selection_for_binary_classification import convert_to_onnx as convert_fs_to_onnx
from ..latent_variables.latent_variables_for_binary_classification import convert_to_onnx as convert_latent_to_onnx
from ..heads.head_for_binary_classification import convert_to_onnx as convert_head_to_onnx

import onnx
from onnx import compose
from onnx import helper


NUM_CPU = max(1, int(multiprocessing.cpu_count() / 2))


class BaseDefaultBinaryClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, preprocessor_params=None, feature_selector_params=None, latent_params=None, head_params=None):
        self.preprocessor_params = preprocessor_params
        self.feature_selector_params = feature_selector_params
        self.latent_params = latent_params
        self.head_params = head_params

    def find_params(self, X, y):
        if self.preprocessor_params is None:
            logger.info("Finding preprocessor parameters...")
            self.preprocessor_params = find_preprocessor_params(X)
        X = Preprocessor(**self.preprocessor_params).fit(X).transform(X)
        if self.feature_selector_params is None:
            logger.info("Finding feature selector parameters...")
            self.feature_selector_params = find_feature_selector_params(X, y)
        X = FeatureSelectorForBinaryClassification(**self.feature_selector_params).fit(X, y).transform(X)
        if self.latent_params is None:
            logger.info("Finding latent variable parameters...")
            self.latent_params = find_latent_params(X, y)
        X = LatentVariablesForBinaryClassification(**self.latent_params).fit(X, y).transform(X)
        if self.head_params is None:
            logger.info("Finding head parameters...")
            self.head_params = find_head_params(X, y)
        logger.info("Found parameters:")
        logger.info(f"Preprocessor params: {self.preprocessor_params}")
        logger.info(f"Feature selector params: {self.feature_selector_params}")
        logger.info(f"Latent params: {self.latent_params}")
        logger.info(f"Head params: {self.head_params}")
        return self
    
    def get_params(self):
        return {
            "preprocessor_params": self.preprocessor_params,
            "feature_selector_params": self.feature_selector_params,
            "latent_params": self.latent_params,
            "head_params": self.head_params,
        }
    
    def clear_params(self):
        self.preprocessor_params = None
        self.feature_selector_params = None
        self.latent_params = None
        self.head_params = None
        
    def fit(self, X, y):
        if self.preprocessor_params is None or self.feature_selector_params is None or self.latent_params is None or self.head_params is None:
            self.find_params(X, y)
        logger.info("Fitting preprocessor...")
        self.preprocessor = Preprocessor(**self.preprocessor_params)
        self.preprocessor.fit(X)
        X = self.preprocessor.transform(X)
        logger.info("Fitting feature selector...")
        self.feature_selector = FeatureSelectorForBinaryClassification(**self.feature_selector_params)
        self.feature_selector.fit(X, y)
        X = self.feature_selector.transform(X)
        logger.info("Fitting latent variable reducer...")
        self.latent_reducer = LatentVariablesForBinaryClassification(**self.latent_params)
        self.latent_reducer.fit(X, y)
        X = self.latent_reducer.transform(X)
        self.head = HeadForBinaryClassification(**self.head_params)
        self.head.fit(X, y)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "preprocessor") or self.preprocessor is None:
            raise ValueError("Model not fitted. Call `fit` first.")
        X = self.preprocessor.transform(X)
        X = self.feature_selector.transform(X)
        X = self.latent_reducer.transform(X)
        return self.head.predict_proba(X)

    def save(self, model_dir: str):
        self.preprocessor.save(model_dir)
        self.feature_selector.save(model_dir)
        self.latent_reducer.save(model_dir)
        self.head.save(model_dir)

    @classmethod
    def load(cls, model_dir: str):
        obj = cls()
        obj.preprocessor = Preprocessor.load(model_dir)
        obj.feature_selector = FeatureSelectorForBinaryClassification.load(model_dir)
        obj.latent_reducer = LatentVariablesForBinaryClassification.load(model_dir)
        obj.head = HeadForBinaryClassification.load(model_dir)
        return obj



class LazyDefaultBinaryClassifier(object):
    def __init__(
        self,
        num_trials: int = 5,
        base_test_size: float = 0.25,
        base_num_splits: int = 3,
        min_positive_proportion: float = 0.01,
        max_positive_proportion: float = 0.5,
        min_samples: int = 30,
        max_samples: int = None,
        min_positive_samples: int = 10,
        max_num_partitions: int = 100,
        min_seen_across_partitions: int = None,
        force_max_positive_proportion_at_partition: bool = False,
        force_on_disk: bool = False,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.base_test_size = base_test_size
        self.base_num_splits = base_num_splits
        self.base_num_trials = num_trials
        self.min_positive_proportion = min_positive_proportion
        self.max_positive_proportion = max_positive_proportion
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.min_positive_samples = min_positive_samples
        self.max_num_partitions = max_num_partitions
        self.min_seen_across_partitions = min_seen_across_partitions
        self.force_max_positive_proportion_at_partition = (
            force_max_positive_proportion_at_partition
        )
        self.force_on_disk = force_on_disk
        self.fit_time = None
        self.models = None
        self.indices = None

    def fit(self, X=None, y=None, h5_file=None, h5_idxs=None):
        t0 = time.time()
        iu = InputUtils()
        su = SamplingUtils()
        iu.evaluate_input(
            X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=y, is_y_mandatory=True
        )
        X, h5_file, h5_idxs = iu.preprocessing(
            X=X, h5_file=h5_file, h5_idxs=h5_idxs, force_on_disk=self.force_on_disk
        )
        if self.max_samples is None:
            self.max_samples = BinaryClassifierMaxSamplesDecider(
                X=X,
                y=y,
                min_samples=self.min_samples,
                min_positive_proportion=self.min_positive_proportion,
            ).decide()
            logger.debug(f"Decided to use max samples: {self.max_samples}")
        if self.min_seen_across_partitions is None:
            theoretical_min = su.get_theoretical_min_seen(y, self.max_samples)
            min_seen_across_partitions = max(1, theoretical_min)
            self.min_seen_across_partitions = min(min_seen_across_partitions, 3)
        models = []
        params = []
        for idxs in su.get_partition_indices(
            X=X,
            h5_file=h5_file,
            h5_idxs=h5_idxs,
            y=y,
            min_positive_proportion=self.min_positive_proportion,
            max_positive_proportion=self.max_positive_proportion,
            min_samples=self.min_samples,
            max_samples=self.max_samples,
            min_positive_samples=self.min_positive_samples,
            max_num_partitions=self.max_num_partitions,
            min_seen_across_partitions=self.min_seen_across_partitions,
            force_max_positive_proportion_at_partition=self.force_max_positive_proportion_at_partition,
        ):
            if h5_file is not None:
                with h5py.File(h5_file, "r") as f:
                    keys = f.keys()
                    if "values" in keys:
                        values_key = "values"
                    elif "Values" in keys:
                        values_key = "Values"
                    else:
                        raise Exception("HDF5 does not contain a values key")
                    X_sampled = iu.h5_data_reader(
                        f[values_key], [h5_idxs[i] for i in idxs]
                    )
            else:
                X_sampled = X[idxs]
            y_sampled = y[idxs]
            logger.debug(
                f"Fitting model on {len(idxs)} samples, positive samples: {np.sum(y_sampled)}, negative samples: {len(y_sampled) - np.sum(y_sampled)}, number of features {X_sampled.shape[1]}"
            )
            if len(params) < 3:
                model = BaseDefaultBinaryClassifier()
                model.find_params(X_sampled, y_sampled)
                params_ = model.get_params()
                params += [params_]
                model.fit(X_sampled, y_sampled)
            else:
                idxs = [i for i in range(len(params))]
                params_ = params[random.choice(idxs)]
                model = BaseDefaultBinaryClassifier(
                    preprocessor_params=params_["preprocessor_params"],
                    feature_selection_params=params_["feature_selector_params"],
                    latent_params=params_["latent_params"],
                    head_params=params_["head_params"],
                )
                model.fit(X_sampled, y_sampled)
            logger.info("Model has successfull been fitted!")
            models += [model]
        self.models = models
        t1 = time.time()
        self.fit_time = t1 - t0
        logger.info(f"Fitting completed in {self.fit_time:.2f} seconds.")
        return self

    def predict(self, X=None, h5_file=None, h5_idxs=None, chunk_size=1000):
        iu = InputUtils()
        iu.evaluate_input(
            X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=None, is_y_mandatory=False
        )
        X, h5_file, h5_idxs = iu.preprocessing(
            X=X, h5_file=h5_file, h5_idxs=h5_idxs, force_on_disk=self.force_on_disk
        )
        su = SamplingUtils()
        if self.models is None:
            raise Exception("No models fitted yet.")
        y_hat = []
        for model in self.models:
            if h5_file is None:
                n = X.shape[0]
                y_hat_ = []
                for X_chunk in tqdm(
                    su.chunk_matrix(X, chunk_size), desc="Predicting chunks..."
                ):
                    y_hat_ += list(model.predict_proba(X_chunk)[:, 1])
            else:
                n = len(h5_idxs)
                y_hat_ = []
                for X_chunk in tqdm(
                    su.chunk_h5_file(h5_file, h5_idxs, chunk_size),
                    desc="Predicting chunks...",
                ):
                    y_hat_ += list(model.predict_proba(X_chunk)[:, 1])
            y_hat += [y_hat_]
        y_hat = np.array(y_hat).T
        y_hat = np.mean(y_hat, axis=1)
        assert len(y_hat) == n, (
            "Predicted labels length does not match input samples length."
        )
        return y_hat

    def save(self, model_dir: str):
        if os.path.exists(model_dir):
            logger.debug(f"Model directory already exists: {model_dir}, deleting it...")
            shutil.rmtree(model_dir)
        logger.debug(f"Creating model directory: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        if self.models is None:
            raise Exception("No models fitted yet.")
        partition_idx = 0
        for model in self.models:
            suffix = str(partition_idx).zfill(3)
            partition_dir = os.path.join(model_dir, f"partition_{suffix}")
            os.makedirs(partition_dir, exist_ok=True)
            logger.debug(f"Saving model to {partition_dir}")
            model.save(partition_dir)
            partition_idx += 1
        metadata = {
            "num_partitions": len(self.models),
            "random_state": self.random_state,
            "base_test_size": self.base_test_size,
            "base_num_splits": self.base_num_splits,
            "base_num_trials": self.base_num_trials,
            "fit_time": self.fit_time,
        }
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def load(cls, model_dir: str):
        obj = cls()
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise Exception("Metadata file not found.")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        obj.random_state = metadata.get("random_state", None)
        obj.base_test_size = metadata.get("base_test_size", None)
        obj.base_num_splits = metadata.get("base_num_splits", None)
        obj.base_num_trials = metadata.get("base_num_trials", None)
        obj.fit_time = metadata.get("fit_time", None)
        num_partitions = metadata.get("num_partitions", None)
        if num_partitions <= 0:
            raise Exception("No partitions found in metadata.")
        obj.models = []
        for i in range(num_partitions):
            suffix = str(i).zfill(3)
            partition_dir = os.path.join(model_dir, f"partition_{suffix}")
            logger.debug(f"Loading model from {partition_dir}")
            model = BaseDefaultBinaryClassifier.load(partition_dir)
            obj.models += [model]
        return obj
    

def _onnx_logger(model):
    logger.info("**** ONNX Model Details ****")
    logger.info(f"ONNX model: {model.graph.name} (ir_version: {model.ir_version}, opset_import: {[opset.version for opset in model.opset_import]})")
    for node in model.graph.node:
        logger.info(f"  Node: {node.name} (op_type: {node.op_type}, inputs: {node.input}, outputs: {node.output})")
        for input_tensor in model.graph.input:
            dims = [d.dim_value if (d.HasField("dim_value")) else "?" for d in input_tensor.type.tensor_type.shape.dim]
            logger.info(f"    Input: {input_tensor.name}, shape: {dims}")
        for output_tensor in model.graph.output:
            dims = [d.dim_value if (d.HasField("dim_value")) else "?" for d in output_tensor.type.tensor_type.shape.dim]
            logger.info(f"    Output: {output_tensor.name}, shape: {dims}")
    logger.info("****************************")


def _check_graph_outputs(model):
    g = model.graph
    produced = set()
    for node in g.node:
        produced.update(node.output)
    declared_outputs = [out.name for out in g.output]
    logger.info(f"Model: {g.name}")
    for out in declared_outputs:
        if out in produced:
            logger.info(f"✅ Graph output '{out}' is produced by a node.")
            return True
        else:
            logger.info(f"❌ Graph output '{out}' is NOT produced by any node!")
            return False

def _fix_graph_outputs_with_identity(model):
    if _check_graph_outputs(model):
        return model
    g = model.graph
    suffix = g.name.lower()
    if g.output:
        out_name = g.output[0].name
        last_node_out = g.node[-1].output[0]
        if out_name != last_node_out:
            identity_node = helper.make_node(
                "Identity",
                inputs=[last_node_out],
                outputs=[out_name],
                name=f"OutputFixer_{suffix}"
            )
            g.node.append(identity_node)
    return model


def convert_partition_to_onnx(partition_dir: str):
    model_dir = partition_dir
    preprocess_onnx_file = convert_preprocess_to_onnx(model_dir)
    fs_onnx_file = convert_fs_to_onnx(model_dir)
    latent_onnx_file = convert_latent_to_onnx(model_dir)
    head_onnx_file = convert_head_to_onnx(model_dir)
    onnx_graphs = []
    for onnx_file in [preprocess_onnx_file, fs_onnx_file, latent_onnx_file, head_onnx_file]:
        if onnx_file is None:
            continue
        onnx_graphs += [onnx.load(onnx_file)]

    onnx_graphs = [_fix_graph_outputs_with_identity(m) for m in onnx_graphs]

    for onnx_model in onnx_graphs:
        _onnx_logger(onnx_model)

    model = onnx_graphs[0]
    for next_model in onnx_graphs[1:]:
        logger.debug(next_model.graph.name)
        src_output = model.graph.output[0].name
        dst_input = next_model.graph.input[0].name
        model = compose.merge_models(model, next_model, io_map=[(src_output, dst_input)])

    final_onnx_path = os.path.join(partition_dir, "lazy_model.onnx")
    onnx.save(model, final_onnx_path)
    return final_onnx_path
