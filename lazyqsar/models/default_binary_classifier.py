import json
import os
import random
import shutil
import time

import h5py
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin

from ..preprocess import prep
from ..feature_selection.binary_classification import fs
from ..latent_variables.binary_classification import lv
from ..heads.binary_classification import mlp, lr, svc

from ..utils.deciders import BinaryClassifierMaxSamplesDecider
from ..utils.io import InputUtils
from ..utils.samplers import BinaryClassifierSamplingUtils as SamplingUtils

from ..utils.logging import logger

import onnx
from onnx import compose
from onnx import helper


class BaseDefaultBinaryClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, params: dict = None):
        if params is None:
            params = {}
        logger.info("Initializing BaseDefaultBinaryClassifier...")
        self.prep_params = params.get("prep", None)
        self.fs_params = params.get("fs", None)
        self.lv_params = params.get("lv", None)
        self.lr_params = params.get("lr", None)
        self.svc_params = params.get("svc", None)
        self.fs_lr_params = params.get("fs_lr", None)
        self.fs_svc_params = params.get("fs_svc", None)
        self.lv_lr_params = params.get("lv_lr", None)
        self.lv_svc_params = params.get("lv_svc", None)
        self.lv_mlp_params = params.get("lv_mlp", None)

    def find_params(self, X, y):
        if self.prep_params is None:
            logger.info("Finding preprocessor parameters...")
            self.prep_params = prep.find_params(X)
        X = prep.Preprocessor(**self.prep_params).fit(X).transform(X)
        if self.fs_params is None:
            logger.info("Finding feature selector parameters...")
            self.fs_params = fs.find_params(X, y)
        X_fs = fs.FeatureSelector(**self.fs_params).fit(X, y).transform(X)
        if self.lv_params is None:
            logger.info("Finding latent variable parameters...")
            self.lv_params = lv.find_params(X, y)
        X_lv = lv.LatentVariables(**self.lv_params).fit(X, y).transform(X)
        if self.lr_params is None:
            logger.info("Finding raw head LR parameters...")
            self.lr_params = lr.find_params(X, y)
        if self.svc_params is None:
            logger.info("Finding raw head SVC parameters...")
            self.svc_params = svc.find_params(X, y)
        if self.fs_lr_params is None:
            logger.info("Finding feature selection with head LR parameters...")
            self.fs_lr_params = lr.find_params(X_fs, y)
        if self.fs_svc_params is None:
            logger.info("Finding feature selection with head SVC parameters...")
            self.fs_svc_params = svc.find_params(X_fs, y)
        if self.lv_lr_params is None:
            logger.info("Finding latent variables with head LR parameters...")
            self.lv_lr_params = lr.find_params(X_lv, y)
        if self.lv_svc_params is None:
            logger.info("Finding latent variables with head SVC parameters...")
            self.lv_svc_params = svc.find_params(X_lv, y)
        if self.lv_mlp_params is None:
            logger.info("Finding latent variables with head MLP parameters...")
            self.lv_mlp_params = mlp.find_params(X_lv, y)
        return self
    
    def get_params(self):
        return {
            "prep_params": self.prep_params,
            "fs_params": self.fs_params,
            "lv_params": self.lv_params,
            "lr_params": self.lr_params,
            "svc_params": self.svc_params,
            "fs_lr_params": self.fs_lr_params,
            "fs_svc_params": self.fs_svc_params,
            "lv_lr_params": self.lv_lr_params,
            "lv_svc_params": self.lv_svc_params,
            "lv_mlp_params": self.lv_mlp_params,            
        }
    
    def clear_params(self):
        self.prep_params = None
        self.fs_params = None
        self.lv_params = None
        self.lr_params = None
        self.svc_params = None
        self.fs_lr_params = None
        self.fs_svc_params = None
        self.lv_lr_params = None
        self.lv_svc_params = None
        self.lv_mlp_params = None
        
    def fit(self, X, y):
        if self.prep_params is None:
            self.find_params(X, y)
        logger.info("Fitting preprocessor...")
        self.prep = prep.Preprocessor(**self.prep_params)
        self.prep.fit(X)
        X = self.prep.transform(X)
        logger.info("Fitting feature selector...")
        self.fs = fs.FeatureSelector(**self.fs_params)
        self.fs.fit(X, y)
        X_fs = self.fs.transform(X)
        logger.info("Fitting latent variable reducer...")
        self.lv = lv.LatentVariables(**self.lv_params)
        self.lv.fit(X, y)
        X_lv = self.lv.transform(X)
        logger.info("Fitting heads...")
        self.lr = lr.Head(**self.lr_params).fit(X, y)
        self.svc = svc.Head(**self.svc_params).fit(X, y)
        self.fs_lr = lr.Head(**self.fs_lr_params).fit(X_fs, y)
        self.fs_svc = svc.Head(**self.fs_svc_params).fit(X_fs, y)
        self.lv_lr = lr.Head(**self.lv_lr_params).fit(X_lv, y)
        self.lv_svc = svc.Head(**self.lv_svc_params).fit(X_lv, y)
        self.lv_mlp = mlp.Head(**self.lv_mlp_params).fit(X_lv, y)
        logger.info("Fitting completed")
        self.model_names = ["lr", "svc", "fs_lr", "fs_svc", "lv_lr", "lv_svc", "lv_mlp"]
        self.model_scores = [
            self.lr.score,
            self.svc.score,
            self.fs_lr.score,
            self.fs_svc.score,
            self.lv_lr.score,
            self.lv_svc.score,
            self.lv_mlp.score,
        ]
        self.weights = np.clip(np.array(self.model_scores) - 0.5, 0, 1)
        self.weights = self.weights / np.sum(self.weights)
        logger.info(f"Individual model scores: {self.model_scores}")
        logger.info(f"Model weights: {self.weights}")
        return self

    def predict_proba(self, X):
        logger.debug("Predicting probabilities")
        X = self.prep.transform(X)
        X_fs = self.fs.transform(X)
        X_lv = self.lv.transform(X)
        y_lr = self.lr.predict_proba(X)[:, 1]
        y_svc = self.svc.predict_proba(X)[:, 1]
        y_fs_lr = self.fs_lr.predict_proba(X_fs)[:, 1]
        y_fs_svc = self.fs_svc.predict_proba(X_fs)[:, 1]
        y_lv_lr = self.lv_lr.predict_proba(X_lv)[:, 1]
        y_lv_svc = self.lv_svc.predict_proba(X_lv)[:, 1]
        y_lv_mlp = self.lv_mlp.predict_proba(X_lv)[:, 1]
        y_hat = np.array([
            y_lr,
            y_svc,
            y_fs_lr,
            y_fs_svc,
            y_lv_lr,
            y_lv_svc,
            y_lv_mlp]).T
        y_hat = np.average(y_hat, axis=1, weights=self.weights)
        return np.vstack([1 - y_hat, y_hat]).T

    def save(self, model_dir: str):
        self.prep.save("prep", model_dir)
        self.fs.save("fs", model_dir)
        self.lv.save("lv", model_dir)
        self.lr.save("lr", model_dir)
        self.svc.save("svc", model_dir)
        self.fs_lr.save("fs_lr", model_dir)
        self.fs_svc.save("fs_svc", model_dir)
        self.lv_lr.save("lv_lr", model_dir)
        self.lv_svc.save("lv_svc", model_dir)
        self.lv_mlp.save("lv_mlp", model_dir)
        metadata = {
            "prep_params": self.prep_params,
            "fs_params": self.fs_params,
            "lv_params": self.lv_params,
            "lr_params": self.lr_params,
            "svc_params": self.svc_params,
            "fs_lr_params": self.fs_lr_params,
            "fs_svc_params": self.fs_svc_params,
            "lv_lr_params": self.lv_lr_params,
            "lv_svc_params": self.lv_svc_params,
            "lv_mlp_params": self.lv_mlp_params,
            "model_names": self.model_names,
            "model_scores": self.model_scores,
            "weights": self.weights.tolist()}
        metadata_path = os.path.join(model_dir, "metadata.json")
        logger.info("Saving metadata to {0}".format(metadata_path))
        metadata["prep_params"] = bool(metadata["prep_params"]["is_sparse"])
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def load(cls, model_dir: str):
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        params = {
            "prep": metadata.get("prep_params", None),
            "fs": metadata.get("fs_params", None),
            "lv": metadata.get("lv_params", None),
            "lr": metadata.get("lr_params", None),
            "svc": metadata.get("svc_params", None),
            "fs_lr": metadata.get("fs_lr_params", None),
            "fs_svc": metadata.get("fs_svc_params", None),
            "lv_lr": metadata.get("lv_lr_params", None),
            "lv_svc": metadata.get("lv_svc_params", None),
            "lv_mlp": metadata.get("lv_mlp_params", None),
        }
        obj = cls(params)
        obj.prep = prep.Preprocessor.load("prep", model_dir)
        obj.fs = fs.FeatureSelector.load("fs", model_dir)
        obj.lv = lv.LatentVariables.load("lv", model_dir)
        obj.lr = lr.Head.load("lr", model_dir)
        obj.svc = svc.Head.load("svc", model_dir)
        obj.fs_lr = lr.Head.load("fs_lr", model_dir)
        obj.fs_svc = svc.Head.load("fs_svc", model_dir)
        obj.lv_lr = lr.Head.load("lv_lr", model_dir)
        obj.lv_svc = svc.Head.load("lv_svc", model_dir)
        obj.lv_mlp = mlp.Head.load("lv_mlp", model_dir)
        obj.scores = metadata.get("model_scores", None)
        obj.model_names = metadata.get("model_names", None)
        obj.weights = np.array(metadata.get("weights", None))
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
                    params=params_
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
            logger.info(f"Graph output '{out}' is produced by a node.")
            return True
        else:
            logger.info(f"Graph output '{out}' is not produced by any node")
            return False

def _fix_graph_outputs_with_identity(model):
    g = model.graph
    name = g.name.lower()
    if not _check_graph_outputs(model):
        if g.output:
            out_name = g.output[0].name
            last_node_out = g.node[-1].output[0]
            if out_name != last_node_out:
                identity_node = helper.make_node(
                    "Identity",
                    inputs=[last_node_out],
                    outputs=[out_name],
                    name=f"OutputFixer_{name}"
                )
                g.node.append(identity_node)
    model = compose.add_prefix(model, f"{name}_")
    return model


def convert_partition_to_onnx(partition_dir: str):
    model_dir = partition_dir
    prep_onnx_file = prep.convert_to_onnx("prep", model_dir)
    fs_onnx_file = fs.convert_to_onnx("fs", model_dir)
    lv_onnx_file = lv.convert_to_onnx("lv", model_dir)
    lr_onnx_file = lr.convert_to_onnx("lr", model_dir)
    svc_onnx_file = svc.convert_to_onnx("svc", model_dir)
    fs_lr_onnx_file = lr.convert_to_onnx("fs_lr", model_dir)
    fs_svc_onnx_file = svc.convert_to_onnx("fs_svc", model_dir)
    lv_lr_onnx_file = lr.convert_to_onnx("lv_lr", model_dir)
    lv_svc_onnx_file = svc.convert_to_onnx("lv_svc", model_dir)
    lv_mlp_onnx_file = mlp.convert_to_onnx("lv_mlp", model_dir)
    onnx_graphs = {}
    onnx_graphs["prep"] = onnx.load(prep_onnx_file)
    onnx_graphs["fs"] = onnx.load(fs_onnx_file)
    onnx_graphs["lv"] = onnx.load(lv_onnx_file)
    onnx_graphs["lr"] = onnx.load(lr_onnx_file)
    onnx_graphs["svc"] = onnx.load(svc_onnx_file)
    onnx_graphs["fs_lr"] = onnx.load(fs_lr_onnx_file)
    onnx_graphs["fs_svc"] = onnx.load(fs_svc_onnx_file)
    onnx_graphs["lv_lr"] = onnx.load(lv_lr_onnx_file)
    onnx_graphs["lv_svc"] = onnx.load(lv_svc_onnx_file)
    onnx_graphs["lv_mlp"] = onnx.load(lv_mlp_onnx_file)
    onnx_graphs = dict((k, _fix_graph_outputs_with_identity(v)) for k, v in onnx_graphs.items())

    for name, onnx_model in onnx_graphs.items():
        logger.info(f"Checking ONNX graph outputs for model: {name}")
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
