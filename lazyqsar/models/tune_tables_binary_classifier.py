
import time
import h5py
import numpy as np
import os
import json
import shutil
from tqdm import tqdm
from tunetables_light.scripts.transformer_prediction_interface import TuneTablesClassifierLight as TuneTablesClassifierLightBase
from tunetables_light.scripts.transformer_prediction_interface import TuneTablesZeroShotClassifier as TuneTablesZeroShotClassifierBase
from .utils import BinaryClassifierSamplingUtils as SamplingUtils
from .utils import InputUtils


class BaseTuneTablesBinaryClassifier(TuneTablesClassifierLightBase):
    def __init__(
            self, 
            device="cpu",
            epoch=4,
            batch_size=4,
            lr=0.1,
            dropout=0.2,
            tuned_prompt_size=5,
            early_stopping=10,
            boosting=False,
            bagging=False, 
            ensemble_size=5,
            average_ensemble=False,
            positive_fraction=0.2
            ):
        super().__init__(
            lr=lr, 
            epoch=epoch, 
            device=device, 
            tuned_prompt_size=tuned_prompt_size, 
            batch_size=batch_size,
            dropout=dropout,
            early_stopping=early_stopping,
            boosting=boosting,
            bagging=bagging,
            ensemble_size=ensemble_size,
            average_ensemble=average_ensemble,
            positive_fraction=positive_fraction
        )
    def predict(self, X):
        return super().predict_proba(X)[:, 1]

class BaseTuneTablesZeroShotBinaryClassifier(TuneTablesZeroShotClassifierBase):
    def __init__(self, subsample_features=True):
        super().__init__(subsample_features=subsample_features)



class LazyTuneTablesBinaryClassifier(object):
    def __init__(self,
                 min_positive_proportion: float = 0.01,
                 max_positive_proportion: float = 0.5,
                 min_samples: int = 30,
                 max_samples: int = 10000,
                 min_positive_samples: int = 10,
                 max_num_partitions: int = 100,
                 min_seen_across_partitions: int = 1,
                 force_on_disk: bool = False):
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
        self.model = BaseTuneTablesBinaryClassifier()

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
        self.model.fit(X, y)
        t1 = time.time()
        self.fit_time = t1 - t0
        print(f"Fitting completed in {self.fit_time:.2f} seconds.")
        return self
    
    def predict(self, X=None, h5_file=None, h5_idxs=None, chunk_size=1000):
        chunk_size = X.shape[0]

        iu = InputUtils()
        iu.evaluate_input(X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=None, is_y_mandatory=False)
        X, h5_file, h5_idxs = iu.preprocessing(X=X, h5_file=h5_file, h5_idxs=h5_idxs, force_on_disk=self.force_on_disk)
        su = SamplingUtils()
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Please call fit() before predict().")
        model = self.model
        y_hat = []
        n = X.shape[0]
        if h5_file is None:
            y_hat = list(model.predict(X))
        else:
            for X_chunk in tqdm(su.chunk_h5_file(h5_file, h5_idxs, chunk_size), desc="Predicting chunks..."):
                print(f"X chunk vales: {X_chunk}")
                y_hat += list(model.predict(X_chunk))
        y_hat = np.array(y_hat)
        assert len(y_hat) == n, f"Predicted labels length does not match input samples length. Y pred: {len(y_hat)} and sample size: {n}"
        return y_hat

    def save_model(self, model_dir: str):
        if os.path.exists(model_dir):
            print(f"Model directory already exists: {model_dir}, deleting it...")
            shutil.rmtree(model_dir)
        print(f"Creating model directory: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        if self.model is None:
            raise Exception("No model fitted yet.")
        self.model.save_model(model_dir)

    @classmethod
    def load_model(cls, model_dir: str):
        obj = cls()
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise Exception("Metadata file not found.")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        obj.fit_time = metadata.get("fit_time", None)
        obj.model = BaseTuneTablesBinaryClassifier.load_model(model_dir)
        return obj
