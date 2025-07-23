import os
import h5py
import numpy as np
import psutil


class InputUtils(object):
    def __init__(self):
        pass

    def evaluate_input(
        self, X=None, h5_file=None, h5_idxs=None, y=None, is_y_mandatory=True
    ):
        if is_y_mandatory:
            if y is None:
                raise ValueError("y cannot be None. Provide a label vector.")
        if X is None and h5_file is None:
            raise ValueError("Either X or h5_file must be provided.")
        if X is not None and h5_file is not None:
            raise ValueError("Provide either X or h5_file, not both.")
        if h5_file is not None:
            if not os.path.exists(h5_file):
                raise FileNotFoundError(f"File {h5_file} does not exist.")
            if not h5_file.endswith(".h5"):
                raise ValueError("h5_file should be a .h5 file.")
            if h5_idxs is None:
                with h5py.File(h5_file, "r") as f:
                    keys = f.keys()
                    if "values" in keys:
                        values_key = "values"
                    elif "Values" in keys:
                        values_key = "Values"
                    else:
                        raise Exception("HDF5 does not contain a values key")
                    h5_idxs = [i for i in range(f[values_key].shape[0])]
            else:
                if y is not None:
                    if len(h5_idxs) != len(y):
                        raise Exception("h5_idxs length must match y length.")
        if X is not None and h5_idxs is not None:
            raise Exception(
                "You cannot provide h5_idxs if X is provided. Use X only or h5_file with h5_idxs."
            )

    def h5_data_reader(self, x_data, idxs):
        sorted_indices = np.argsort(idxs)
        sorted_idxs = np.array(idxs)[sorted_indices]
        sorted_data = x_data[sorted_idxs, :]
        inverse_sort = np.argsort(sorted_indices)
        x = sorted_data[inverse_sort]
        return x

    def is_load_full_h5_file(self, h5_file):
        with h5py.File(h5_file, "r") as f:
            keys = f.keys()
            if "values" in keys:
                values_key = "values"
            elif "Values" in keys:
                values_key = "Values"
            else:
                raise Exception("HDF5 does not contain a values key")
            dataset = f[values_key]
            if isinstance(dataset, h5py.Dataset):
                size_bytes = dataset.size * dataset.dtype.itemsize
                size_gb = size_bytes / (1024**3)
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        print(
            f"Available memory: {available_gb:.2f} GB, H5 file size: {size_gb:.2f} GB"
        )
        if available_gb > size_gb * 1.5:
            return True
        else:
            return False

    def preprocessing(self, X=None, h5_file=None, h5_idxs=None, force_on_disk=False):
        if h5_file is not None:
            if h5_idxs is None:
                with h5py.File(h5_file, "r") as f:
                    keys = f.keys()
                    if "values" in keys:
                        values_key = "values"
                    elif "Values" in keys:
                        values_key = "Values"
                    else:
                        raise Exception("HDF5 does not contain a values key")
                    h5_idxs = [i for i in range(f[values_key].shape[0])]
            if not force_on_disk and self.is_load_full_h5_file(h5_file):
                print("Loading full h5 file into memory...")
                with h5py.File(h5_file, "r") as f:
                    keys = f.keys()
                    if "values" in keys:
                        values_key = "values"
                    elif "Values" in keys:
                        values_key = "Values"
                    else:
                        raise Exception("HDF5 does not contain a values key")
                    X = f[values_key][:]
                    X = X[h5_idxs, :]
                    h5_file = None
                    h5_idxs = None
        return X, h5_file, h5_idxs