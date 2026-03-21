import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass

import h5py
import joblib
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

from zspreprocessing import ZeroShotClassifierPreprocessor

from .. import ONNX_IR_VERSION
from ..heads.binary_classification import lr
try:
    from ..heads.binary_classification import xgb as xgb_head
    _XGB_HEAD_AVAILABLE = True
except ImportError:
    xgb_head = None
    _XGB_HEAD_AVAILABLE = False

from ..utils.io import InputUtils
from ..utils.samplers import BinaryClassifierSamplingUtils as SamplingUtils, compute_head_weights

from ..utils.logging import logger

import onnx
from onnx import compose
from onnx import helper
from onnx import TensorProto


ALL_HEADS = ["lr", "xgb"]

HEAD_MODULES = {
    "lr": lr, "xgb": xgb_head,
}

# "prep_variable" is the output tensor name from zspreprocessing ONNX after
# compose.add_prefix(model, "prep_") (skl2onnx names transformer outputs "variable")
HEAD_ONNX_INPUT = {
    "lr": "prep_variable", "xgb": "prep_variable",
}

HEAD_FAMILY = {
    "lr": "linear",
    "xgb": "boosting",
}

PROFILE_HEADS = {
    ("tiny",   False): ["lr", "xgb"],
    ("tiny",   True):  ["lr", "xgb"],
    ("small",  False): ["lr", "xgb"],
    ("small",  True):  ["lr", "xgb"],
    ("medium", False): ["lr", "xgb"],
    ("medium", True):  ["lr", "xgb"],
    ("large",  False): ["lr", "xgb"],
    ("large",  True):  ["lr", "xgb"],
}

if not _XGB_HEAD_AVAILABLE:
    ALL_HEADS = [h for h in ALL_HEADS if h != "xgb"]
    HEAD_MODULES.pop("xgb", None)
    HEAD_ONNX_INPUT.pop("xgb", None)
    HEAD_FAMILY.pop("xgb", None)
    PROFILE_HEADS = {k: [h for h in v if h != "xgb"] for k, v in PROFILE_HEADS.items()}

PROFILE_CAPS = {"tiny": 2, "small": 3, "medium": 4, "large": 3}


@dataclass
class ShapePolicy:
    profile: str = "small"
    n_samples: int = 0
    minority_count: int = 0
    n_features_after_prep: int = 0
    is_sparse: bool = False
    feature_sample_ratio: float = 0.0
    candidate_heads: tuple = ()
    max_heads: int = 3


def _matrix_density(X) -> float:
    if sparse.issparse(X):
        return float(X.nnz) / float(np.prod(X.shape))
    return 1.0 - float(np.mean(np.asarray(X) == 0))


def _profile_from_shape(n_samples: int, minority_count: int) -> str:
    if n_samples < 300 or minority_count < 30:
        return "tiny"
    if n_samples < 2_000:
        return "small"
    if n_samples < 20_000:
        return "medium"
    return "large"


def derive_shape_policy(
    X,
    y,
    is_sparse: bool | None = None,
    max_heads: int | None = None,
):
    y = np.asarray(y, dtype=int)
    n_samples = int(len(y))
    minority_count = int(min(np.sum(y == 1), np.sum(y == 0)))
    n_features = int(X.shape[1])
    if is_sparse is None:
        is_sparse = bool(sparse.issparse(X) or _matrix_density(X) <= 0.1)
    feature_sample_ratio = float(n_features / max(n_samples, 1))
    profile = _profile_from_shape(n_samples, minority_count)
    candidate_heads = list(PROFILE_HEADS[(profile, is_sparse)])

    default_max_heads = PROFILE_CAPS[profile]

    return ShapePolicy(
        profile=profile,
        n_samples=n_samples,
        minority_count=minority_count,
        n_features_after_prep=n_features,
        is_sparse=is_sparse,
        feature_sample_ratio=feature_sample_ratio,
        candidate_heads=tuple(candidate_heads),
        max_heads=min(max_heads or default_max_heads, len(candidate_heads)),
    )


def _clip_probs(y_prob):
    return np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)


HEAD_PARAMS_ATTR = {"lr": "lr_params", "xgb": "xgb_params"}


class BaseEclecticBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params: dict = None):
        if params is None:
            params = {}
        logger.info("Initializing BaseEclecticBinaryClassifier...")

        self.prep_params = params.get("prep", None)

        self.lr_params = params.get("lr", None)
        self.xgb_params = params.get("xgb", None)

        self.active_heads = params.get("active_heads", None)
        self.max_heads = params.get("max_heads", None)
        self.shape_policy = params.get("shape_policy", None)
        self._fit_cache = None

    def find_params(self, X, y):
        logger.info("Fitting ZeroShotClassifierPreprocessor...")
        preprocessor = ZeroShotClassifierPreprocessor()
        X = preprocessor.fit_transform(X, y)

        self.prep_params = {
            "scaler_name": preprocessor.scaler_name_,
            "reducer_name": preprocessor.reducer_name_,
            "n_features_in": preprocessor.n_features_in_,
            "n_features_out": preprocessor.n_features_out_,
        }

        is_sparse = bool(preprocessor.profile_.sparsity > 0.1 or preprocessor.profile_.is_sparse_counts)

        if self.shape_policy is None:
            self.shape_policy = asdict(
                derive_shape_policy(
                    X,
                    y,
                    is_sparse=is_sparse,
                    max_heads=self.max_heads,
                )
            )
        shape_policy = ShapePolicy(**self.shape_policy)
        if self.active_heads is None:
            self.active_heads = list(shape_policy.candidate_heads)
        logger.info(
            f"Shape policy: profile={shape_policy.profile}, sparse={shape_policy.is_sparse}, "
            f"features={shape_policy.n_features_after_prep}, candidates={self.active_heads}"
        )

        for name in self.active_heads:
            attr = HEAD_PARAMS_ATTR[name]
            if getattr(self, attr) is None:
                logger.info(f"Finding parameters for head: {name}")
                setattr(self, attr, HEAD_MODULES[name].find_params(X, y))

        self._fit_cache = {"prep": preprocessor, "X": X}
        return self

    def get_params(self):
        return {
            "prep": self.prep_params,
            "lr": self.lr_params,
            "xgb": self.xgb_params,
            "active_heads": self.active_heads,
            "max_heads": self.max_heads,
            "shape_policy": self.shape_policy,
        }

    def clear_params(self):
        self.prep_params = None
        self.lr_params = None
        self.xgb_params = None
        self.shape_policy = None

    def fit(self, X, y):
        if self.prep_params is None:
            self.find_params(X, y)

        if self._fit_cache is not None:
            logger.info("Reusing fitted preprocessor from find_params...")
            self.prep = self._fit_cache["prep"]
            X = self._fit_cache["X"]
            self._fit_cache = None
        else:
            logger.info("Fitting ZeroShotClassifierPreprocessor...")
            self.prep = ZeroShotClassifierPreprocessor()
            X = self.prep.fit_transform(X, y)

        if self.shape_policy is None:
            is_sparse = bool(self.prep.profile_.sparsity > 0.1 or self.prep.profile_.is_sparse_counts)
            self.shape_policy = asdict(
                derive_shape_policy(
                    X,
                    y,
                    is_sparse=is_sparse,
                    max_heads=self.max_heads,
                )
            )

        if self.active_heads is None:
            shape_policy = ShapePolicy(**self.shape_policy)
            self.active_heads = list(shape_policy.candidate_heads)

        logger.info(f"Fitting heads: {self.active_heads}")
        for name in ALL_HEADS:
            if name not in self.active_heads:
                setattr(self, name, None)

        for name in self.active_heads:
            params = getattr(self, HEAD_PARAMS_ATTR[name])
            setattr(self, name, HEAD_MODULES[name].Head(**params).fit(X, y))

        self.model_names = list(self.active_heads)
        self.model_scores = [0.5] * len(self.model_names)
        self.weights = np.ones(len(self.model_names), dtype=float) / len(self.model_names)
        return self

    def predict_proba(self, X):
        logger.debug("Predicting probabilities")
        X = self.prep.transform(X)
        y_hats = [
            getattr(self, n).predict_proba(X)[:, 1]
            for n in self.model_names
        ]
        y_hat = np.average(np.array(y_hats).T, axis=1, weights=self.weights)
        return np.vstack([1 - y_hat, y_hat]).T

    def save(self, model_dir: str):
        joblib.dump(self.prep, os.path.join(model_dir, "prep.joblib"))

        for name in self.active_heads:
            getattr(self, name).save(name, model_dir)

        metadata = {
            "prep_params": self.prep_params,
            "lr_params": self.lr_params,
            "xgb_params": self.xgb_params,
            "active_heads": self.active_heads,
            "model_names": self.model_names,
            "model_scores": self.model_scores,
            "weights": self.weights.tolist(),
            "max_heads": self.max_heads,
            "shape_policy": self.shape_policy,
        }
        metadata_path = os.path.join(model_dir, "metadata.json")
        logger.info("Saving metadata to {0}".format(metadata_path))
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def load(cls, model_dir: str):
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        params = {
            "prep": metadata.get("prep_params", None),
            "lr": metadata.get("lr_params", None),
            "xgb": metadata.get("xgb_params", None),
            "max_heads": metadata.get("max_heads", None),
            "shape_policy": metadata.get("shape_policy", None),
        }

        obj = cls(params)
        obj.prep = joblib.load(os.path.join(model_dir, "prep.joblib"))

        active_heads = metadata.get("active_heads", ALL_HEADS)
        for name in ALL_HEADS:
            if name in active_heads:
                setattr(obj, name, HEAD_MODULES[name].Head.load(name, model_dir))
            else:
                setattr(obj, name, None)

        obj.active_heads = active_heads
        obj.model_scores = metadata.get("model_scores", None)
        obj.model_names = metadata.get("model_names", None)
        obj.weights = np.array(metadata.get("weights", None))

        return obj


class LazyEclecticBinaryClassifier(object):
    def __init__(
        self,
        max_heads: int | None = None,
        max_samples: int = 100_000,
        max_num_partitions: int = 100,
        force_on_disk: bool = False,
        random_state: int = 42,
    ):
        self.max_heads = max_heads
        self.random_state = random_state
        self.max_samples = max_samples
        self.max_num_partitions = max_num_partitions
        self.force_on_disk = force_on_disk
        self.fit_time = None
        self.models = None
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
        models = []
        params = None
        for idxs in su.get_partition_indices(
            X=X,
            h5_file=h5_file,
            h5_idxs=h5_idxs,
            y=y,
            max_num_partitions=self.max_num_partitions,
            max_samples=self.max_samples,
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
            if not params:
                model = BaseEclecticBinaryClassifier(
                    {
                        "max_heads": self.max_heads,
                    }
                )
                model.find_params(X_sampled, y_sampled)
                params = model.get_params()
                model.fit(X_sampled, y_sampled)
            else:
                model = BaseEclecticBinaryClassifier(params=params)
                model.fit(X_sampled, y_sampled)
            logger.info("Model has been fitted successfully.")
            models += [model]
        self.models = models
        self.score = float(np.mean([np.mean(m.model_scores) for m in self.models]))
        t1 = time.time()
        self.fit_time = t1 - t0
        logger.info(f"Fitting completed in {self.fit_time:.2f} seconds.")
        return self

    def predict(self, X=None, h5_file=None, h5_idxs=None, chunk_size=10_000):
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
                logger.debug(
                    f"Predicting on {n} samples with chunk size {chunk_size}..."
                )
                for X_chunk in su.chunk_matrix(X, chunk_size):
                    y_hat_ += list(model.predict_proba(X_chunk)[:, 1])
                    logger.debug(f"Predicted {len(y_hat_)} samples so far...")
            else:
                n = len(h5_idxs)
                y_hat_ = []
                logger.debug(
                    f"Predicting on {n} samples from HDF5 with chunk size {chunk_size}..."
                )
                for X_chunk in su.chunk_h5_file(h5_file, h5_idxs, chunk_size):
                    y_hat_ += list(model.predict_proba(X_chunk)[:, 1])
                    logger.debug(f"Predicted {len(y_hat_)} samples so far...")
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
            "max_heads": self.max_heads,
            "random_state": self.random_state,
            "fit_time": self.fit_time,
            "score": float(np.mean([np.mean(m.model_scores) for m in self.models])),
        }
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def load(cls, model_dir: str):
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise Exception("Metadata file not found.")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        obj = cls(
            max_heads=metadata.get("max_heads", None),
        )
        obj.random_state = metadata.get("random_state", None)
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
        logger.debug("**** ONNX Model Details ****")
        logger.debug(
            f"ONNX model: {model.graph.name} (ir_version: {model.ir_version}, opset_import: {[opset.version for opset in model.opset_import]})"
        )
        for node in model.graph.node:
            logger.debug(
                f"  Node: {node.name} (op_type: {node.op_type}, inputs: {list(node.input)}, outputs: {list(node.output)})"
            )
        for input_tensor in model.graph.input:
            dims = [
                d.dim_value
                if d.HasField("dim_value")
                else (d.dim_param if d.HasField("dim_param") else "?")
                for d in input_tensor.type.tensor_type.shape.dim
            ]
            logger.debug(f"    Input: {input_tensor.name}, shape: {dims}")
        for output_tensor in model.graph.output:
            dims = [
                d.dim_value
                if d.HasField("dim_value")
                else (d.dim_param if d.HasField("dim_param") else "?")
                for d in output_tensor.type.tensor_type.shape.dim
            ]
            logger.debug(f"    Output: {output_tensor.name}, shape: {dims}")
        logger.debug("****************************")

    def _check_graph_outputs(model):
        g = model.graph
        produced = {o for n in g.node for o in n.output}
        ok = True
        logger.debug(f"Model: {g.name}")
        for out in [o.name for o in g.output]:
            if out in produced:
                logger.debug(f"Graph output '{out}' is produced by a node.")
            else:
                logger.warning(f"Graph output '{out}' is not produced by any node in '{g.name}'")
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

    metadata_path = os.path.join(partition_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    active_heads = metadata.get("active_heads", ALL_HEADS)
    model_names = metadata.get("model_names", active_heads)

    # Convert preprocessor: load from joblib, export to ONNX
    prep_joblib_path = os.path.join(model_dir, "prep.joblib")
    preprocessor = joblib.load(prep_joblib_path)
    prep_onnx_path = os.path.join(model_dir, "prep.onnx")
    preprocessor.to_onnx(prep_onnx_path)

    prep_onnx_model = onnx.load(prep_onnx_path)
    # Set graph name to "prep" so that after add_prefix("prep_") the output
    # tensor "variable" becomes "prep_variable" (predictable tensor name)
    prep_onnx_model.graph.name = "prep"
    onnx.save(prep_onnx_model, prep_onnx_path)

    onnx_graphs = {
        "prep": onnx.load(prep_onnx_path),
    }

    # Convert active heads
    for name in active_heads:
        head_file = HEAD_MODULES[name].convert_to_onnx(name, model_dir)
        onnx_graphs[name] = onnx.load(head_file)

    # Normalize IR version and opset across all models before merging.
    # Collect the union of all extra domains (e.g. ai.onnx.ml) across models
    # so that all merged models declare the same opset set.
    from lazyqsar import ONNX_TARGET_OPSET
    extra_domains: dict[str, int] = {}
    for model in onnx_graphs.values():
        for op in model.opset_import:
            if op.domain != "":
                extra_domains[op.domain] = max(extra_domains.get(op.domain, 0), op.version)

    for key in list(onnx_graphs.keys()):
        model = onnx_graphs[key]
        # Replace opset_import: one entry per domain, no duplicates
        del model.opset_import[:]
        model.opset_import.append(helper.make_opsetid("", ONNX_TARGET_OPSET))
        for domain, version in extra_domains.items():
            model.opset_import.append(helper.make_opsetid(domain, version))
        model.ir_version = ONNX_IR_VERSION

    onnx_graphs = {k: _fix_graph_outputs_with_identity(v) for k, v in onnx_graphs.items()}
    onnx_graphs = {k: densify(v) for k, v in onnx_graphs.items()}

    for name, onnx_model in onnx_graphs.items():
        logger.debug(f"Checking ONNX graph outputs for model: {name}")
        _onnx_logger(onnx_model)

    logger.info("Merging ONNX graphs...")
    # After _fix_graph_outputs_with_identity with graph name "prep",
    # the preprocessor output tensor is named "prep_variable"
    prep_output_tensor = "prep_variable"
    accumulated_outputs = [prep_output_tensor]
    model = onnx_graphs["prep"]

    for name in active_heads:
        head_output = f"{name}_output_{name}"
        accumulated_outputs.append(head_output)
        model = compose.merge_models(
            model, onnx_graphs[name],
            io_map=[(prep_output_tensor, f"{name}_input_{name}")],
            outputs=list(accumulated_outputs),
        )

    head_outputs = [f"{name}_output_{name}" for name in model_names]
    weights = np.array(metadata.get("weights", None), dtype=np.float32)
    if weights is None or len(weights) != len(head_outputs):
        logger.warning("Weights missing or wrong length; using uniform weights.")
        weights = np.ones(len(head_outputs), dtype=np.float32) / float(
            len(head_outputs)
        )
    logger.debug(f"Weights: {weights}")

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
