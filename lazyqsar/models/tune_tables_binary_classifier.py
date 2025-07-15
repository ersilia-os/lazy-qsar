
import time
import h5py
import numpy as np
import os
import json
import shutil
from tqdm import tqdm

from .utils import SamplingUtils, InputUtils

class BaseTuneTablesBinaryClassifier(object):
    # TODO class




class TuneTablesBinaryClassifier(object):

    def __init__(self,
                 reducer_method: str = None,
                 max_reducer_dim: int = 500,
                 num_trials: int = 10,
                 base_test_size: float = 0.25,
                 base_num_splits: int = 3,
                 base_timeout: int = 120,
                 min_positive_proportion: float = 0.01,
                 max_positive_proportion: float = 0.5,
                 min_samples: int = 30,
                 max_samples: int = 10000,
                 min_positive_samples: int = 10,
                 max_num_partitions: int = 100,
                 min_seen_across_partitions: int = 1,
                 force_on_disk: bool = False,
                 random_state: int = 42):
        self.reducer_method = reducer_method
        self.max_reducer_dim = max_reducer_dim
        self.random_state = random_state
        self.base_test_size = base_test_size
        self.base_num_splits = base_num_splits
        self.base_num_trials = num_trials
        self.base_timeout = base_timeout
        self.min_positive_proportion = min_positive_proportion
        self.max_positive_proportion = max_positive_proportion
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.min_positive_samples = min_positive_samples
        self.max_num_partitions = max_num_partitions
        self.min_seen_across_partitions = min_seen_across_partitions
        self.force_on_disk = force_on_disk
        self.fit_time = None
        self.indices = None
        self.model = None

    def fit(self, X=None, y=None, h5_file=None, h5_idxs=None):
        t0 = time.time()
        iu = InputUtils()
        iu.evaluate_input(X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=y, is_y_mandatory=True)
        X, h5_file, h5_idxs = iu.preprocessing(X=X, h5_file=h5_file, h5_idxs=h5_idxs, force_on_disk=self.force_on_disk)
        if h5_file is not None:
            with h5py.File(h5_file, "r") as f:
                X = iu.h5_data_reader(f["values"], [idx for idx in h5_idxs])
        else:
            pass
        # TODO Fit model
        # self.model = BaseTuneTablesBinaryClassifier(...)
        # self.model.fit(X, y)
        t1 = time.time()
        self.fit_time = t1 - t0
        print(f"Fitting completed in {self.fit_time:.2f} seconds.")
        return self

    def predict(self, X=None, h5_file=None, h5_idxs=None, chunk_size=1000):
        iu = InputUtils()
        iu.evaluate_input(X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=None, is_y_mandatory=False)
        X, h5_file, h5_idxs = iu.preprocessing(X=X, h5_file=h5_file, h5_idxs=h5_idxs, force_on_disk=self.force_on_disk)
        su = SamplingUtils()
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Please call fit() before predict().")
        model = self.model
        y_hat = []
        if h5_file is None:
            n = X.shape[0]
            for X_chunk in tqdm(su.chunk_matrix(X, chunk_size), desc="Predicting chunks..."):
                y_hat += list(model.predict_proba(X_chunk)[:,1])
        else:
            n = len(h5_idxs)
            for X_chunk in tqdm(su.chunk_h5_file(h5_file, h5_idxs, chunk_size), desc="Predicting chunks..."):
                y_hat += list(model.predict_proba(X_chunk)[:,1])
        y_hat = np.array(y_hat)
        assert len(y_hat) == n, "Predicted labels length does not match input samples length."
        return y_hat

    def save_model(self, model_dir: str):
        if os.path.exists(model_dir):
            print(f"Model directory already exists: {model_dir}, deleting it...")
            shutil.rmtree(model_dir)
        print(f"Creating model directory: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        if self.model is None:
            raise Exception("No model fitted yet.")
        # TODO save model
        # self.model.save_model(model_dir)
        metadata = {
            "num_partitions": len(self.model),
            "reducer_method": self.reducer_method,
            "max_reducer_dim": self.max_reducer_dim,
            "random_state": self.random_state,
            "base_test_size": self.base_test_size,
            "base_num_splits": self.base_num_splits,
            "base_num_trials": self.base_num_trials,
            "base_timeout": self.base_timeout,
            "fit_time": self.fit_time
        }
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def load_model(cls, model_dir: str):
        obj = cls()
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise Exception("Metadata file not found.")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        obj.reducer_method = metadata.get("reducer_method", None)
        obj.max_reducer_dim = metadata.get("max_reducer_dim", None)
        obj.random_state = metadata.get("random_state", None)
        obj.base_test_size = metadata.get("base_test_size", None)
        obj.base_num_splits = metadata.get("base_num_splits", None)
        obj.base_num_trials = metadata.get("base_num_trials", None)
        obj.base_timeout = metadata.get("base_timeout", None)
        obj.fit_time = metadata.get("fit_time", None)
        num_partitions = metadata.get("num_partitions", None)
        if num_partitions <= 0:
            raise Exception("No partitions found in metadata.")
        # TODO load model
        # obj.model = BaseTuneTablesBinaryClassifier.load_model(model_dir)
        return obj