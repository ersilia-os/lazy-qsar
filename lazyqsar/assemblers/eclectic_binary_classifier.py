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
from ..feature_selection.binary_classification import fs, mfs
from ..latent_variables.binary_classification import lv
from ..heads.binary_classification import mlp, lr, svc, et

from ..utils.deciders import BinaryClassifierMaxSamplesDecider
from ..utils.io import InputUtils
from ..utils.samplers import BinaryClassifierSamplingUtils as SamplingUtils

from ..utils.logging import logger

import onnx
from onnx import compose
from onnx import helper
from onnx import TensorProto


NUM_TRIALS = 10


class BaseEclecticBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params: dict = None):
        if params is None:
            params = {}
        logger.info("Initializing BaseEclecticBinaryClassifier...")

        self.prep_params = params.get("prep", None)

        self.fs_params = params.get("fs", None)
        self.mfs_params = params.get("mfs", None)
        self.lv_params = params.get("lv", None)

        self.lr_params = params.get("lr", None)
        self.svc_params = params.get("svc", None)
        self.et_params = params.get("et", None)

        self.fs_lr_params = params.get("fs_lr", None)
        self.fs_svc_params = params.get("fs_svc", None)

        self.mfs_lr_params = params.get("mfs_lr", None)
        self.mfs_svc_params = params.get("mfs_svc", None)

        self.lv_lr_params = params.get("lv_lr", None)
        self.lv_svc_params = params.get("lv_svc", None)
        self.lv_mlp_params = params.get("lv_mlp", None)

        self.num_trials = NUM_TRIALS

    def find_params(self, X, y, num_trials):
        if self.prep_params is None:
            logger.info("Finding preprocessor parameters...")
            self.prep_params = prep.find_params(X)

        X = prep.Preprocessor(**self.prep_params).fit(X).transform(X)

        if self.fs_params is None:
            logger.info("Finding feature selector parameters...")
            self.fs_params = fs.find_params(X, y, num_trials)
        X_fs = fs.FeatureSelector(**self.fs_params).fit(X, y).transform(X)
        if self.mfs_params is None:
            logger.info("Finding model feature selector parameters...")
            self.mfs_params = mfs.find_params(X, y, num_trials)
        X_mfs = mfs.FeatureSelector(**self.mfs_params).fit(X, y).transform(X)
        if self.lv_params is None:
            logger.info("Finding latent variable parameters...")
            self.lv_params = lv.find_params(X, y, num_trials)
        X_lv = lv.LatentVariables(**self.lv_params).fit(X, y).transform(X)

        if self.lr_params is None:
            logger.info("Finding raw head LR parameters...")
            self.lr_params = lr.find_params(X, y, num_trials)
        if self.svc_params is None:
            logger.info("Finding raw head SVC parameters...")
            self.svc_params = svc.find_params(X, y, num_trials)
        if self.et_params is None:
            logger.info("Finding raw head ET parameters...")
            self.et_params = et.find_params(X, y, num_trials)

        if self.fs_lr_params is None:
            logger.info("Finding feature selection with head LR parameters...")
            self.fs_lr_params = lr.find_params(X_fs, y, num_trials)
        if self.fs_svc_params is None:
            logger.info("Finding feature selection with head SVC parameters...")
            self.fs_svc_params = svc.find_params(X_fs, y, num_trials)
        if self.mfs_lr_params is None:
            logger.info("Finding model feature selection with head LR parameters...")
            self.mfs_lr_params = lr.find_params(X_mfs, y, num_trials)
        if self.mfs_svc_params is None:
            logger.info("Finding model feature selection with head SVC parameters...")
            self.mfs_svc_params = svc.find_params(X_mfs, y, num_trials)
        if self.lv_lr_params is None:
            logger.info("Finding latent variables with head LR parameters...")
            self.lv_lr_params = lr.find_params(X_lv, y, num_trials)
        if self.lv_svc_params is None:
            logger.info("Finding latent variables with head SVC parameters...")
            self.lv_svc_params = svc.find_params(X_lv, y, num_trials)
        if self.lv_mlp_params is None:
            logger.info("Finding latent variables with head MLP parameters...")
            self.lv_mlp_params = mlp.find_params(X_lv, y, num_trials)
        return self

    def get_params(self):
        return {
            "prep_params": self.prep_params,
            "fs_params": self.fs_params,
            "mfs_params": self.mfs_params,
            "lv_params": self.lv_params,
            "lr_params": self.lr_params,
            "svc_params": self.svc_params,
            "et_params": self.et_params,
            "fs_lr_params": self.fs_lr_params,
            "fs_svc_params": self.fs_svc_params,
            "mfs_lr_params": self.mfs_lr_params,
            "mfs_svc_params": self.mfs_svc_params,
            "lv_lr_params": self.lv_lr_params,
            "lv_svc_params": self.lv_svc_params,
            "lv_mlp_params": self.lv_mlp_params,
        }

    def clear_params(self):
        self.prep_params = None

        self.fs_params = None
        self.mfs_params = None
        self.lv_params = None

        self.lr_params = None
        self.svc_params = None
        self.et_params = None

        self.fs_lr_params = None
        self.fs_svc_params = None

        self.mfs_lr_params = None
        self.mfs_svc_params = None

        self.lv_lr_params = None
        self.lv_svc_params = None
        self.lv_mlp_params = None

    def fit(self, X, y):
        if self.prep_params is None:
            self.find_params(X, y, self.num_trials)
        logger.info("Fitting preprocessor...")
        self.prep = prep.Preprocessor(**self.prep_params)
        self.prep.fit(X)
        X = self.prep.transform(X)

        logger.info("Fitting feature selector...")
        self.fs = fs.FeatureSelector(**self.fs_params)
        self.fs.fit(X, y)
        X_fs = self.fs.transform(X)
        logger.info("Fitting model feature selector...")
        self.mfs = mfs.FeatureSelector(**self.mfs_params)
        self.mfs.fit(X, y)
        X_mfs = self.mfs.transform(X)
        logger.info("Fitting latent variable reducer...")
        self.lv = lv.LatentVariables(**self.lv_params)
        self.lv.fit(X, y)
        X_lv = self.lv.transform(X)

        logger.info("Fitting raw heads...")
        self.lr = lr.Head(**self.lr_params).fit(X, y)
        self.svc = svc.Head(**self.svc_params).fit(X, y)
        self.et = et.Head(**self.et_params).fit(X, y)

        logger.info("Fitting feature selection heads...")
        self.fs_lr = lr.Head(**self.fs_lr_params).fit(X_fs, y)
        self.fs_svc = svc.Head(**self.fs_svc_params).fit(X_fs, y)

        logger.info("Fitting model feature selection heads...")
        self.mfs_lr = lr.Head(**self.mfs_lr_params).fit(X_mfs, y)
        self.mfs_svc = svc.Head(**self.mfs_svc_params).fit(X_mfs, y)

        logger.info("Fitting latent variable heads...")
        self.lv_lr = lr.Head(**self.lv_lr_params).fit(X_lv, y)
        self.lv_svc = svc.Head(**self.lv_svc_params).fit(X_lv, y)
        self.lv_mlp = mlp.Head(**self.lv_mlp_params).fit(X_lv, y)

        logger.info("Fitting completed")
        self.model_names = [
            "lr",
            "svc",
            "et",
            "fs_lr",
            "fs_svc",
            "mfs_lr",
            "mfs_svc",
            "lv_lr",
            "lv_svc",
            "lv_mlp",
        ]
        self.model_scores = [
            self.lr.score,
            self.svc.score,
            self.et.score,
            self.fs_lr.score,
            self.fs_svc.score,
            self.mfs_lr.score,
            self.mfs_svc.score,
            self.lv_lr.score,
            self.lv_svc.score,
            self.lv_mlp.score,
        ]
        self.weights = np.clip(np.array(self.model_scores) - 0.5, 0, 1) + 1e-4
        self.weights = self.weights / np.sum(self.weights)
        logger.info(f"Individual model scores: {self.model_scores}")
        logger.info(f"Model weights: {self.weights}")
        return self

    def predict_proba(self, X):
        logger.debug("Predicting probabilities")
        X = self.prep.transform(X)

        X_fs = self.fs.transform(X)
        X_mfs = self.mfs.transform(X)
        X_lv = self.lv.transform(X)

        y_lr = self.lr.predict_proba(X)[:, 1]
        y_svc = self.svc.predict_proba(X)[:, 1]
        y_et = self.et.predict_proba(X)[:, 1]

        y_fs_lr = self.fs_lr.predict_proba(X_fs)[:, 1]
        y_fs_svc = self.fs_svc.predict_proba(X_fs)[:, 1]

        y_mfs_lr = self.mfs_lr.predict_proba(X_mfs)[:, 1]
        y_mfs_svc = self.mfs_svc.predict_proba(X_mfs)[:, 1]

        y_lv_lr = self.lv_lr.predict_proba(X_lv)[:, 1]
        y_lv_svc = self.lv_svc.predict_proba(X_lv)[:, 1]
        y_lv_mlp = self.lv_mlp.predict_proba(X_lv)[:, 1]

        y_hat = np.array(
            [
                y_lr,
                y_svc,
                y_et,
                y_fs_lr,
                y_fs_svc,
                y_mfs_lr,
                y_mfs_svc,
                y_lv_lr,
                y_lv_svc,
                y_lv_mlp,
            ]
        ).T
        y_hat = np.average(y_hat, axis=1, weights=self.weights)
        return np.vstack([1 - y_hat, y_hat]).T

    def save(self, model_dir: str):
        self.prep.save("prep", model_dir)

        self.fs.save("fs", model_dir)
        self.mfs.save("mfs", model_dir)
        self.lv.save("lv", model_dir)

        self.lr.save("lr", model_dir)
        self.svc.save("svc", model_dir)
        self.et.save("et", model_dir)

        self.fs_lr.save("fs_lr", model_dir)
        self.fs_svc.save("fs_svc", model_dir)

        self.mfs_lr.save("mfs_lr", model_dir)
        self.mfs_svc.save("mfs_svc", model_dir)

        self.lv_lr.save("lv_lr", model_dir)
        self.lv_svc.save("lv_svc", model_dir)
        self.lv_mlp.save("lv_mlp", model_dir)

        metadata = {
            "prep_params": self.prep_params,
            "fs_params": self.fs_params,
            "mfs_params": self.mfs_params,
            "lv_params": self.lv_params,
            "lr_params": self.lr_params,
            "svc_params": self.svc_params,
            "et_params": self.et_params,
            "fs_lr_params": self.fs_lr_params,
            "fs_svc_params": self.fs_svc_params,
            "mfs_lr_params": self.mfs_lr_params,
            "mfs_svc_params": self.mfs_svc_params,
            "lv_lr_params": self.lv_lr_params,
            "lv_svc_params": self.lv_svc_params,
            "lv_mlp_params": self.lv_mlp_params,
            "model_names": self.model_names,
            "model_scores": self.model_scores,
            "weights": self.weights.tolist(),
            "num_trials": self.num_trials,
        }
        metadata_path = os.path.join(model_dir, "metadata.json")
        logger.info("Saving metadata to {0}".format(metadata_path))
        metadata["prep_params"]["is_sparse"] = bool(
            metadata["prep_params"]["is_sparse"]
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def load(cls, model_dir: str):
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        params = {
            "prep": metadata.get("prep_params", None),
            "fs": metadata.get("fs_params", None),
            "mfs": metadata.get("mfs_params", None),
            "lv": metadata.get("lv_params", None),
            "lr": metadata.get("lr_params", None),
            "svc": metadata.get("svc_params", None),
            "et": metadata.get("et_params", None),
            "fs_lr": metadata.get("fs_lr_params", None),
            "fs_svc": metadata.get("fs_svc_params", None),
            "mfs_lr": metadata.get("mfs_lr_params", None),
            "mfs_svc": metadata.get("mfs_svc_params", None),
            "lv_lr": metadata.get("lv_lr_params", None),
            "lv_svc": metadata.get("lv_svc_params", None),
            "lv_mlp": metadata.get("lv_mlp_params", None),
        }

        obj = cls(params)
        obj.prep = prep.Preprocessor.load("prep", model_dir)

        obj.fs = fs.FeatureSelector.load("fs", model_dir)
        obj.mfs = mfs.FeatureSelector.load("mfs", model_dir)
        obj.lv = lv.LatentVariables.load("lv", model_dir)

        obj.lr = lr.Head.load("lr", model_dir)
        obj.svc = svc.Head.load("svc", model_dir)
        obj.et = et.Head.load("et", model_dir)

        obj.fs_lr = lr.Head.load("fs_lr", model_dir)
        obj.fs_svc = svc.Head.load("fs_svc", model_dir)

        obj.mfs_lr = lr.Head.load("mfs_lr", model_dir)
        obj.mfs_svc = svc.Head.load("mfs_svc", model_dir)

        obj.lv_lr = lr.Head.load("lv_lr", model_dir)
        obj.lv_svc = svc.Head.load("lv_svc", model_dir)
        obj.lv_mlp = mlp.Head.load("lv_mlp", model_dir)

        obj.model_scores = metadata.get("model_scores", None)
        obj.model_names = metadata.get("model_names", None)
        obj.weights = np.array(metadata.get("weights", None))
        obj.num_trials = metadata.get("num_trials", None)

        return obj


class LazyEclecticBinaryClassifier(object):
    def __init__(
        self,
        num_trials: int = 10,
        min_positive_proportion: float = 0.01,
        max_positive_proportion: float = 0.5,
        min_samples: int = 30,
        max_samples: int = 10000,
        min_positive_samples: int = 10,
        max_num_partitions: int = 100,
        min_seen_across_partitions: int = 1,
        force_max_positive_proportion_at_partition: bool = False,
        force_on_disk: bool = False,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.num_trials = num_trials
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
        self.score = None

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
                model = BaseEclecticBinaryClassifier()
                model.num_trials = self.num_trials
                model.find_params(X_sampled, y_sampled, self.num_trials)
                params_ = model.get_params()
                params += [params_]
                model.fit(X_sampled, y_sampled)
            else:
                idxs = [i for i in range(len(params))]
                params_ = params[random.choice(idxs)]
                model = BaseEclecticBinaryClassifier(params=params_)
                model.num_trials = self.num_trials
                model.fit(X_sampled, y_sampled)
            logger.info("Model has successfull been fitted!")
            models += [model]
        self.models = models
        self.score = float(np.mean([np.mean(m.model_scores) for m in self.models]))
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
            "num_trials": self.num_trials,
            "fit_time": self.fit_time,
            "score": float(np.mean([np.mean(m.model_scores) for m in self.models])),
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
        obj.num_trials = metadata.get("num_trials", None)
        obj.fit_time = metadata.get("fit_time", None)
        obj.score = metadata.get("score", None)
        num_partitions = metadata.get("num_partitions", None)
        if num_partitions <= 0:
            raise Exception("No partitions found in metadata.")
        obj.models = []
        for i in range(num_partitions):
            suffix = str(i).zfill(3)
            partition_dir = os.path.join(model_dir, f"partition_{suffix}")
            logger.debug(f"Loading model from {partition_dir}")
            model = BaseEclecticBinaryClassifier.load(partition_dir)
            obj.models += [model]
        return obj


def convert_partition_to_onnx(partition_dir: str, clean: bool = True) -> str:
    if not os.path.exists(partition_dir):
        raise Exception(f"Partition directory does not exist: {partition_dir}")

    if os.path.exists(os.path.join(partition_dir, "lazy_model.onnx")):
        logger.info(
            f"ONNX model already exists in {partition_dir}, skipping conversion."
        )
        return os.path.join(partition_dir, "lazy_model.onnx")

    def _onnx_logger(model):
        logger.info("**** ONNX Model Details ****")
        logger.info(
            f"ONNX model: {model.graph.name} (ir_version: {model.ir_version}, opset_import: {[opset.version for opset in model.opset_import]})"
        )
        for node in model.graph.node:
            logger.info(
                f"  Node: {node.name} (op_type: {node.op_type}, inputs: {list(node.input)}, outputs: {list(node.output)})"
            )
        for input_tensor in model.graph.input:
            dims = [
                d.dim_value
                if d.HasField("dim_value")
                else (d.dim_param if d.HasField("dim_param") else "?")
                for d in input_tensor.type.tensor_type.shape.dim
            ]
            logger.info(f"    Input: {input_tensor.name}, shape: {dims}")
        for output_tensor in model.graph.output:
            dims = [
                d.dim_value
                if d.HasField("dim_value")
                else (d.dim_param if d.HasField("dim_param") else "?")
                for d in output_tensor.type.tensor_type.shape.dim
            ]
            logger.info(f"    Output: {output_tensor.name}, shape: {dims}")
        logger.info("****************************")

    def _check_graph_outputs(model):
        g = model.graph
        produced = {o for n in g.node for o in n.output}
        ok = True
        logger.info(f"Model: {g.name}")
        for out in [o.name for o in g.output]:
            if out in produced:
                logger.info(f"Graph output '{out}' is produced by a node.")
            else:
                logger.info(f"Graph output '{out}' is not produced by any node")
                ok = False
        return ok

    def _fix_graph_outputs_with_identity(model):
        g = model.graph
        name = g.name.lower()
        if not _check_graph_outputs(model):
            if g.output:
                out_name = g.output[0].name
                last_node_out = g.node[-1].output[0]
                if out_name != last_node_out:
                    g.node.append(
                        helper.make_node(
                            "Identity",
                            inputs=[last_node_out],
                            outputs=[out_name],
                            name=f"OutputFixer_{name}",
                        )
                    )

        model = compose.add_prefix(model, f"{name}_")
        return model

    def _standardize_io_names(model, input_name="input", output_name="output"):
        g = model.graph
        assert len(g.input) >= 1, "Merged graph has no external inputs."
        old_in_vi = g.input[0]
        old_in_name = old_in_vi.name

        elem_type = old_in_vi.type.tensor_type.elem_type or TensorProto.FLOAT
        dims = []
        for i, d in enumerate(old_in_vi.type.tensor_type.shape.dim):
            if d.HasField("dim_param"):
                dims.append(d.dim_param if i != 0 else "batch_size")
            elif d.HasField("dim_value"):
                dims.append(d.dim_value if i != 0 else "batch_size")
            else:
                dims.append("batch_size" if i == 0 else None)
        if not dims:
            dims = ["batch_size"]

        g.input.remove(old_in_vi)
        g.input.extend([helper.make_tensor_value_info(input_name, elem_type, dims)])
        g.node.insert(
            0,
            helper.make_node(
                "Identity",
                inputs=[input_name],
                outputs=[old_in_name],
                name="InputAlias",
            ),
        )

        del g.output[:]
        g.output.extend(
            [
                helper.make_tensor_value_info(
                    output_name, TensorProto.FLOAT, ["batch_size"]
                )
            ]
        )
        return model

    def _add_scalar_mul_and_sum(model, weights, head_outputs):
        weighted_inputs = []
        for i, (name_i, wi) in enumerate(zip(head_outputs, weights)):
            w_name = f"w_{i}"
            w_init = helper.make_tensor(
                name=w_name, data_type=onnx.TensorProto.FLOAT, dims=[], vals=[float(wi)]
            )
            model.graph.initializer.append(w_init)
            mul_out = f"weighted_{name_i}"
            weighted_inputs.append(mul_out)
            model.graph.node.append(
                helper.make_node(
                    "Mul",
                    inputs=[name_i, w_name],
                    outputs=[mul_out],
                    name=f"WeightMul_{i}",
                )
            )
        model.graph.node.append(
            helper.make_node(
                "Sum", inputs=weighted_inputs, outputs=["output"], name="WeightedSum"
            )
        )
        return model

    def densify(model):
        if (
            hasattr(model.graph, "sparse_initializer")
            and len(model.graph.sparse_initializer) > 0
        ):
            for si in list(model.graph.sparse_initializer):
                idx = np.array(si.indices.int64_data, dtype=np.int64).reshape(-1, 2)
                vals = np.array(si.values.float_data, dtype=np.float32)
                dense_array = np.zeros(si.dims, dtype=np.float32)
                for (r, c), v in zip(idx, vals):
                    dense_array[r, c] = v
                name = si.values.name or getattr(si, "name", "<unnamed>")
                dense_tensor = onnx.numpy_helper.from_array(dense_array, name=name)
                model.graph.initializer.append(dense_tensor)
            model.graph.ClearField("sparse_initializer")
        return model

    logger.info(f"Converting partition at {partition_dir} to ONNX...")
    model_dir = partition_dir

    prep_onnx_file = prep.convert_to_onnx("prep", model_dir)

    fs_onnx_file = fs.convert_to_onnx("fs", model_dir)
    mfs_onnx_file = mfs.convert_to_onnx("mfs", model_dir)
    lv_onnx_file = lv.convert_to_onnx("lv", model_dir)

    lr_onnx_file = lr.convert_to_onnx("lr", model_dir)
    svc_onnx_file = svc.convert_to_onnx("svc", model_dir)
    et_onnx_file = et.convert_to_onnx("et", model_dir)

    fs_lr_onnx_file = lr.convert_to_onnx("fs_lr", model_dir)
    fs_svc_onnx_file = svc.convert_to_onnx("fs_svc", model_dir)

    mfs_lr_onnx_file = lr.convert_to_onnx("mfs_lr", model_dir)
    mfs_svc_onnx_file = svc.convert_to_onnx("mfs_svc", model_dir)

    lv_lr_onnx_file = lr.convert_to_onnx("lv_lr", model_dir)
    lv_svc_onnx_file = svc.convert_to_onnx("lv_svc", model_dir)
    lv_mlp_onnx_file = mlp.convert_to_onnx("lv_mlp", model_dir)

    onnx_graphs = {
        "prep": onnx.load(prep_onnx_file),
        "fs": onnx.load(fs_onnx_file),
        "mfs": onnx.load(mfs_onnx_file),
        "lv": onnx.load(lv_onnx_file),
        "lr": onnx.load(lr_onnx_file),
        "svc": onnx.load(svc_onnx_file),
        "et": onnx.load(et_onnx_file),
        "fs_lr": onnx.load(fs_lr_onnx_file),
        "fs_svc": onnx.load(fs_svc_onnx_file),
        "mfs_lr": onnx.load(mfs_lr_onnx_file),
        "mfs_svc": onnx.load(mfs_svc_onnx_file),
        "lv_lr": onnx.load(lv_lr_onnx_file),
        "lv_svc": onnx.load(lv_svc_onnx_file),
        "lv_mlp": onnx.load(lv_mlp_onnx_file),
    }
    onnx_graphs = {
        k: _fix_graph_outputs_with_identity(v) for k, v in onnx_graphs.items()
    }

    onnx_graphs = {k: densify(v) for k, v in onnx_graphs.items()}

    for name, onnx_model in onnx_graphs.items():
        logger.info(f"Checking ONNX graph outputs for model: {name}")
        _onnx_logger(onnx_model)

    logger.info("Merging ONNX graphs...")
    model = compose.merge_models(
        onnx_graphs["prep"],
        onnx_graphs["fs"],
        io_map=[("prep_output_prep", "fs_input_fs")],
        outputs=["prep_output_prep", "fs_output_fs"],
    )

    model = compose.merge_models(
        model,
        onnx_graphs["mfs"],
        io_map=[("prep_output_prep", "mfs_input_mfs")],
        outputs=["prep_output_prep", "fs_output_fs", "mfs_output_mfs"],
    )

    model = compose.merge_models(
        model,
        onnx_graphs["lv"],
        io_map=[("prep_output_prep", "lv_input_lv")],
        outputs=["prep_output_prep", "fs_output_fs", "mfs_output_mfs", "lv_output_lv"],
    )

    model = compose.merge_models(
        model,
        onnx_graphs["lr"],
        io_map=[("prep_output_prep", "lr_input_lr")],
        outputs=[
            "prep_output_prep",
            "fs_output_fs",
            "mfs_output_mfs",
            "lv_output_lv",
            "lr_output_lr",
        ],
    )

    model = compose.merge_models(
        model,
        onnx_graphs["svc"],
        io_map=[("prep_output_prep", "svc_input_svc")],
        outputs=[
            "prep_output_prep",
            "fs_output_fs",
            "mfs_output_mfs",
            "lv_output_lv",
            "lr_output_lr",
            "svc_output_svc",
        ],
    )

    model = compose.merge_models(
        model,
        onnx_graphs["et"],
        io_map=[("prep_output_prep", "et_input_et")],
        outputs=[
            "prep_output_prep",
            "fs_output_fs",
            "mfs_output_mfs",
            "lv_output_lv",
            "lr_output_lr",
            "svc_output_svc",
            "et_output_et",
        ],
    )

    model = compose.merge_models(
        model,
        onnx_graphs["fs_lr"],
        io_map=[("fs_output_fs", "fs_lr_input_fs_lr")],
        outputs=[
            "prep_output_prep",
            "fs_output_fs",
            "mfs_output_mfs",
            "lv_output_lv",
            "lr_output_lr",
            "svc_output_svc",
            "et_output_et",
            "fs_lr_output_fs_lr",
        ],
    )
    model = compose.merge_models(
        model,
        onnx_graphs["fs_svc"],
        io_map=[("fs_output_fs", "fs_svc_input_fs_svc")],
        outputs=[
            "prep_output_prep",
            "fs_output_fs",
            "mfs_output_mfs",
            "lv_output_lv",
            "lr_output_lr",
            "svc_output_svc",
            "et_output_et",
            "fs_lr_output_fs_lr",
            "fs_svc_output_fs_svc",
        ],
    )

    model = compose.merge_models(
        model,
        onnx_graphs["mfs_lr"],
        io_map=[("mfs_output_mfs", "mfs_lr_input_mfs_lr")],
        outputs=[
            "prep_output_prep",
            "fs_output_fs",
            "mfs_output_mfs",
            "lv_output_lv",
            "lr_output_lr",
            "svc_output_svc",
            "et_output_et",
            "fs_lr_output_fs_lr",
            "fs_svc_output_fs_svc",
            "mfs_lr_output_mfs_lr",
        ],
    )
    model = compose.merge_models(
        model,
        onnx_graphs["mfs_svc"],
        io_map=[("mfs_output_mfs", "mfs_svc_input_mfs_svc")],
        outputs=[
            "prep_output_prep",
            "fs_output_fs",
            "mfs_output_mfs",
            "lv_output_lv",
            "lr_output_lr",
            "svc_output_svc",
            "et_output_et",
            "fs_lr_output_fs_lr",
            "fs_svc_output_fs_svc",
            "mfs_lr_output_mfs_lr",
            "mfs_svc_output_mfs_svc",
        ],
    )

    model = compose.merge_models(
        model,
        onnx_graphs["lv_lr"],
        io_map=[("lv_output_lv", "lv_lr_input_lv_lr")],
        outputs=[
            "prep_output_prep",
            "fs_output_fs",
            "mfs_output_mfs",
            "lv_output_lv",
            "lr_output_lr",
            "svc_output_svc",
            "et_output_et",
            "fs_lr_output_fs_lr",
            "fs_svc_output_fs_svc",
            "mfs_lr_output_mfs_lr",
            "mfs_svc_output_mfs_svc",
            "lv_lr_output_lv_lr",
        ],
    )

    model = compose.merge_models(
        model,
        onnx_graphs["lv_svc"],
        io_map=[("lv_output_lv", "lv_svc_input_lv_svc")],
        outputs=[
            "prep_output_prep",
            "fs_output_fs",
            "mfs_output_mfs",
            "lv_output_lv",
            "lr_output_lr",
            "svc_output_svc",
            "et_output_et",
            "fs_lr_output_fs_lr",
            "fs_svc_output_fs_svc",
            "mfs_lr_output_mfs_lr",
            "mfs_svc_output_mfs_svc",
            "lv_lr_output_lv_lr",
            "lv_svc_output_lv_svc",
        ],
    )

    model = compose.merge_models(
        model,
        onnx_graphs["lv_mlp"],
        io_map=[("lv_output_lv", "lv_mlp_input_lv_mlp")],
        outputs=[
            "prep_output_prep",
            "fs_output_fs",
            "mfs_output_mfs",
            "lv_output_lv",
            "lr_output_lr",
            "svc_output_svc",
            "et_output_et",
            "fs_lr_output_fs_lr",
            "fs_svc_output_fs_svc",
            "mfs_lr_output_mfs_lr",
            "mfs_svc_output_mfs_svc",
            "lv_lr_output_lv_lr",
            "lv_svc_output_lv_svc",
            "lv_mlp_output_lv_mlp",
        ],
    )

    metadata_path = os.path.join(partition_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    head_outputs = [
        "lr_output_lr",
        "svc_output_svc",
        "et_output_et",
        "fs_lr_output_fs_lr",
        "fs_svc_output_fs_svc",
        "mfs_lr_output_mfs_lr",
        "mfs_svc_output_mfs_svc",
        "lv_lr_output_lv_lr",
        "lv_svc_output_lv_svc",
        "lv_mlp_output_lv_mlp",
    ]
    weights = np.array(metadata.get("weights", None), dtype=np.float32)
    if weights is None or len(weights) != len(head_outputs):
        logger.warning("Weights missing or wrong length; using uniform weights.")
        weights = np.ones(len(head_outputs), dtype=np.float32) / float(
            len(head_outputs)
        )
    logger.info(f"Weights: {weights}")

    model = _add_scalar_mul_and_sum(model, weights, head_outputs)
    model = _standardize_io_names(model, input_name="input", output_name="output")

    final_onnx_path = os.path.join(partition_dir, "lazy_model.onnx")
    onnx.save(model, final_onnx_path)
    logger.info(f"Final FP32 ONNX model saved to {final_onnx_path}")
    _onnx_logger(model)

    if clean:
        logger.info("Cleaning up intermediate files...")
        keep = {"lazy_model.onnx"}
        for fn in os.listdir(partition_dir):
            fp = os.path.join(partition_dir, fn)
            if fn in keep:
                continue
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
            except Exception as e:
                logger.warning(f"Could not remove {fp}: {e}")

    return final_onnx_path


def convert_to_onnx(model_dir: str, clean: bool = True):
    if not os.path.exists(model_dir):
        raise Exception(f"Model directory does not exist: {model_dir}")

    if clean:
        final_path = os.path.join(model_dir, "lazy_model.onnx")
        if os.path.exists(final_path):
            os.remove(final_path)

    logger.info(f"Converting eclectic binary classifier at {model_dir} to ONNX...")
    partitions = []
    for fn in os.listdir(model_dir):
        if fn.startswith("partition_"):
            logger.info(f"Found partition: {fn}")
            partition_dir = os.path.join(model_dir, fn)
            convert_partition_to_onnx(partition_dir, clean=clean)
            partitions.append(partition_dir)

    partition_paths = sorted(os.path.join(p, "lazy_model.onnx") for p in partitions)

    for onnx_file in partition_paths:
        suffix = onnx_file.split("/lazy_model")[0].split("partition_")[-1]
        final_onnx_path = os.path.join(model_dir, f"model_{suffix}.onnx")
        shutil.copy(onnx_file, final_onnx_path)
        logger.info(f"Copied partition ONNX model to {final_onnx_path}")
        shutil.rmtree(onnx_file.split("/lazy_model")[0])
