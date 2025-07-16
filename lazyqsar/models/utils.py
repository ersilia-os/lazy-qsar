import math
import time
import numpy as np
import os
import collections
import random
import h5py
import psutil
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.model_selection import StratifiedKFold



class InputUtils(object):

    def __init__(self):
        pass

    def evaluate_input(self, X=None, h5_file=None, h5_idxs=None, y=None, is_y_mandatory=True):
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
                    if "values" not in f:
                        raise ValueError("h5_file must contain 'values' dataset.")
                    h5_idxs = [i for i in range(f["values"].shape[0])]
            else:
                if y is not None:
                    if len(h5_idxs) != len(y):
                        raise Exception("h5_idxs length must match y length.")
        if X is not None and h5_idxs is not None:
            raise Exception("You cannot provide h5_idxs if X is provided. Use X only or h5_file with h5_idxs.")

    def h5_data_reader(self, x_data, idxs):
        sorted_indices = np.argsort(idxs)
        sorted_idxs = np.array(idxs)[sorted_indices]
        sorted_data = x_data[sorted_idxs, :]
        inverse_sort = np.argsort(sorted_indices)
        x = sorted_data[inverse_sort]
        return x
    
    def is_load_full_h5_file(self, h5_file):
        with h5py.File(h5_file, 'r') as f:
            dataset = f["values"]
            if isinstance(dataset, h5py.Dataset):
                size_bytes = dataset.size * dataset.dtype.itemsize
                size_gb = size_bytes / (1024 ** 3)
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        print(f"Available memory: {available_gb:.2f} GB, H5 file size: {size_gb:.2f} GB")
        if available_gb > size_gb * 1.5:
            return True
        else:
            return False
        
    def preprocessing(self, X=None, h5_file=None, h5_idxs=None, force_on_disk=False):
        if h5_file is not None:
            if h5_idxs is None:
                with h5py.File(h5_file, "r") as f:
                    if "values" not in f:
                        raise ValueError("h5_file must contain 'values' dataset.")
                    h5_idxs = [i for i in range(f["values"].shape[0])]
            if not force_on_disk and self.is_load_full_h5_file(h5_file):
                print("Loading full h5 file into memory...")
                with h5py.File(h5_file, "r") as f:
                    X = f["values"][:]
                    X = X[h5_idxs, :]
                    h5_file = None
                    h5_idxs = None
        return X, h5_file, h5_idxs


class SamplingUtils(object):

    def __init__(self):
        pass

    def chunk_h5_file(self, h5_file, h5_idxs, chunk_size):
        iu = InputUtils()
        with h5py.File(h5_file, "r") as f:
            if "values" not in f:
                raise ValueError("h5_file must contain 'values' dataset.")
            values = f["values"]
            for i in range(0, len(h5_idxs), chunk_size):
                idxs_chunk = h5_idxs[i:i + chunk_size]
                yield iu.h5_data_reader(values, idxs_chunk)

    def chunk_matrix(self, X, chunk_size):
        for i in range(0, X.shape[0], chunk_size):
            yield X[i:i + chunk_size]

    def min_sampling_rounds(self, n_total, n_subsample, target_coverage=0.9):
        n = n_total
        m = n_subsample
        if m <= 0 or n <= 0 or m > n:
            raise ValueError("m must be > 0 and <= n")
        prob_not_seen = 1 - target_coverage
        prob_stay_unseen_per_round = 1 - (m / n)
        if prob_stay_unseen_per_round <= 0:
            return 1
        print(f"Probability of not seeing a sample in one round: {prob_stay_unseen_per_round}")
        rounds = math.log(prob_not_seen) / math.log(prob_stay_unseen_per_round)
        return math.ceil(rounds)
    
    def random_sample(self, idxs, size, idx_counts):
        if len(idxs) < size:
            raise ValueError("Cannot sample more indices than available.")
        idxs = set(idxs)
        count2idx = collections.defaultdict(list)
        for k,v in idx_counts.items():
            if k not in idxs:
                continue
            count2idx[v] += [k]
        counts = sorted(count2idx.keys())
        sampled_idxs = []
        for c in counts:
            v = count2idx[c]
            random.shuffle(v)
            sampled_idxs += v
        return sampled_idxs[:size]

    @staticmethod
    def is_integer_matrix(X):
        X_ = X[:10]
        if np.issubdtype(X_.dtype, np.integer):
            return True
        if np.issubdtype(X_.dtype, int):
            return True
        return np.all(np.equal(X_, np.floor(X_)))

    def quick_auc_estimate(self, X, y):
        if self.is_integer_matrix(X):
            model = BernoulliNB()
        else:
            model = GaussianNB()
        if np.sum(y) == 0:
            raise Exception("No positive samples in the data!!!")
        X = np.array(X)
        y = np.array(y, dtype=int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:,1]
            scores.append(roc_auc_score(y_test, preds))
        return float(np.mean(scores))
    
    def _suggest_row(self, idxs_matrix, accepted_rows, sampled_idxs_counts, min_seen_across_partitions):
        accepted_rows_set = set(accepted_rows)
        insufficient_counts = []
        for k,v in sampled_idxs_counts.items():
            if v < min_seen_across_partitions:
                insufficient_counts += [k]
        if len(insufficient_counts) == 0:
            return None
        insufficient_counts = set(insufficient_counts)
        insufficient_counts_coverage = {}
        for i in range(idxs_matrix.shape[0]):
            if i in accepted_rows_set:
                continue
            r = [int(i) for i in idxs_matrix[i, :]]
            c = 0
            for x in r:
                if x in insufficient_counts:
                    c += 1
            insufficient_counts_coverage[i] = c
        coverages = collections.defaultdict(list)
        for k,v in insufficient_counts_coverage.items():
            coverages[v] += [k]
        max_coverage = max(coverages.keys())
        idxs = sorted(coverages[max_coverage])
        suggested_row = idxs[0]
        return suggested_row
    
    def remove_redundancy_of_sampled_matrix(self, idxs_matrix):
        jaccard_max = 0.99
        remove_rows = []
        for i in range(idxs_matrix.shape[0]):
            idx = idxs_matrix.shape[0] - i - 1
            if idx <= 0:
                continue
            current_row = set([int(x) for x in idxs_matrix[idx, :]])
            for j in range(idxs_matrix.shape[0]):
                if j >= idx:
                    continue
                eval_row = set([int(x) for x in idxs_matrix[j, :]])
                intersection = len(current_row.intersection(eval_row))
                union = len(current_row.union(eval_row))
                if union == 0:
                    jaccard = 0
                else:
                    jaccard = intersection / union
                if jaccard >= jaccard_max:
                    remove_rows += [idx]
        R = []
        for i in range(idxs_matrix.shape[0]):
            if i not in remove_rows:
                R += [idxs_matrix[i, :]]
        idxs_matrix = np.array(R, dtype=int)
        return idxs_matrix
    
    def _get_number_of_positives_and_total(self, n_pos, n_tot, min_positive_proportion, max_positive_proportion, min_samples, max_samples, min_positive_samples):
        if n_pos < min_positive_samples:
            raise Exception(f"Not enough positive samples: {n_pos} < {min_positive_samples}.")
        if n_tot < min_samples:
            raise Exception(f"Not enough total samples: {n_tot} < {min_samples}.")
        n_pos_original = int(n_pos)
        n_tot_original = int(n_tot)
        n_neg_original = n_tot_original - n_pos_original
        print(f"Original positive samples: {n_pos_original}, total samples: {n_tot_original}")
        print("Maximum samples:", max_samples)
        if n_tot <= max_samples:
            pos_prop = n_pos / n_tot
            if pos_prop < min_positive_proportion:
                n_tot = int(np.round(n_pos / min_positive_proportion, 0))
                return n_pos, n_tot
            elif pos_prop > max_positive_proportion:
                n_tot = int(np.round(n_neg_original / (1 - max_positive_proportion), 0))
                n_pos = int(np.round(n_tot * max_positive_proportion, 0))
                if n_pos > n_pos_original:
                    n_pos = n_pos_original
                    n_tot = int(np.round(n_pos / max_positive_proportion, 0))
                else:
                    n_tot = int(np.round(n_pos / max_positive_proportion, 0))
                return n_pos, n_tot
            else:
                return n_pos, n_tot
        else:
            pos_prop_original = n_pos_original / n_tot_original
            if pos_prop_original < min_positive_proportion:
                n_pos = int(np.round(max_samples * min_positive_proportion, 0))
                n_pos = max(n_pos, min_positive_samples)
                n_pos = min(n_pos, n_pos_original)
                pos_prop = n_pos / max_samples
                if pos_prop < min_positive_proportion:
                    n_tot = int(np.round(n_pos / min_positive_proportion, 0))
                else:
                    if n_pos_original / max_samples < max_positive_proportion:
                        n_pos = n_pos_original
                    n_tot = max_samples
                return n_pos, n_tot
            elif pos_prop_original > max_positive_proportion:
                n_pos = int(np.round(max_samples * max_positive_proportion, 0))
                n_tot = max_samples
                return n_pos, n_tot
            else:
                n_pos_check = int(np.round(max_samples * pos_prop_original,0))
                if n_pos_check < min_positive_samples:
                    n_pos = int(np.round(max_samples * min_positive_proportion,0))
                    n_pos = max(n_pos, min_positive_samples)
                    n_tot = max_samples
                    if n_pos_original / n_tot < max_positive_proportion:
                        n_pos = n_pos_original
                    return n_pos, n_tot
                else:
                    n_pos = n_pos_check
                    n_tot = max_samples
                    if n_pos_original / n_tot < max_positive_proportion:
                        n_pos = n_pos_original
                    return n_pos, n_tot

    def get_partition_indices(self,
                              X,
                              h5_file,
                              h5_idxs,
                              y,
                              min_positive_proportion,
                              max_positive_proportion,
                              min_samples,
                              max_samples,
                              min_positive_samples,
                              max_num_partitions,
                              min_seen_across_partitions):
        iu = InputUtils()
        iu.evaluate_input(X=X, h5_file=h5_file, h5_idxs=h5_idxs, y=y, is_y_mandatory=True)
        pos_idxs = [i for i, y_ in enumerate(y) if y_ == 1]
        neg_idxs = [i for i, y_ in enumerate(y) if y_ == 0]
        random.shuffle(pos_idxs)
        random.shuffle(neg_idxs)
        n_tot = len(y)
        n_pos = len(pos_idxs)
        n_neg = len(neg_idxs)
        print(f"Total samples: {n_tot}, positive samples: {n_pos}, negative samples: {n_neg}")
        print(f"Maximum samples per partition: {max_samples}, minimum samples per partition: {min_samples}")
        print(f"Positive proportion: {n_pos / n_tot:.2f}")
        n_pos_samples, n_tot_samples = self._get_number_of_positives_and_total(
            n_pos=n_pos,
            n_tot=n_tot,
            min_positive_proportion=min_positive_proportion,
            max_positive_proportion=max_positive_proportion,
            min_samples=min_samples,
            max_samples=max_samples,
            min_positive_samples=min_positive_samples
        )
        n_neg_samples = n_tot_samples - n_pos_samples
        print(f"Sampling {n_pos_samples} positive and {n_neg_samples} negative samples from {n_tot_samples} total samples.")
        effective_sampling_rounds = 1000
        idxs_list_of_lists = []
        all_idxs = dict((i, 0) for i in range(len(y)))
        patience = 0
        current_size = 0
        max_patience = 10
        for i in tqdm(range(effective_sampling_rounds)):
            pos_idxs_sampled = self.random_sample(pos_idxs, n_pos_samples, all_idxs)
            neg_idxs_sampled = self.random_sample(neg_idxs, n_neg_samples, all_idxs)
            sampled_idxs = pos_idxs_sampled + neg_idxs_sampled
            idxs_list_of_lists += [sorted(sampled_idxs)]
            idxs_matrix = np.array(idxs_list_of_lists, dtype=int)
            idxs_matrix = np.unique(idxs_matrix, axis=0)
            idxs_list_of_lists = []
            for i in range(idxs_matrix.shape[0]):
                r = [int(x) for x in idxs_matrix[i, :]]
                idxs_list_of_lists += [r]
            if current_size == len(idxs_list_of_lists):
                patience += 1
                if patience >= max_patience:
                    break
            else:
                patience = 0
                current_size = len(idxs_list_of_lists)
            all_idxs = dict((i, 0) for i in range(len(y)))
            for r in idxs_list_of_lists:
                for idx in r:
                    all_idxs[idx] += 1
            is_done = True
            for _, v in all_idxs.items():
                if v < min_seen_across_partitions*3:
                    is_done = False
            if is_done:
                print(f"All indices seen at least {min_seen_across_partitions} times. Stopping sampling.")
                break
        idxs_matrix = np.array(idxs_list_of_lists, dtype=int)
        idxs_matrix = np.unique(idxs_list_of_lists, axis=0)
        print(f"Unique sampled indices matrix shape: {idxs_matrix.shape}")
        auc_estimate_timeout = 60
        if h5_file:
            with h5py.File(h5_file, "r") as f:
                auc_estimates = []
                t0 = time.time()
                for i in tqdm(range(idxs_matrix.shape[0])):
                    t1 = time.time()
                    if t1-t0 > auc_estimate_timeout:
                        if len(auc_estimates) > 0:
                            auc_est = float(np.mean(auc_estimates))
                        else:
                            auc_est = 0.5
                    else:
                        idxs_y = idxs_matrix[i, :]
                        idxs_x = [h5_idxs[idx] for idx in idxs_y]
                        X_in = iu.h5_data_reader(f["values"], idxs_x)
                        y_in = [y[idx] for idx in idxs_y]
                        auc_est = self.quick_auc_estimate(X_in, y_in)
                    auc_estimates += [auc_est]
        else:
            auc_estimates = []
            t0 = time.time()
            for i in tqdm(range(idxs_matrix.shape[0])):
                t1 = time.time()
                if t1-t0 > 60:
                    if len(auc_estimates) > auc_estimate_timeout:
                        auc_est = float(np.mean(auc_estimates))
                    else:
                        auc_est = 0.5
                else:
                    idxs_y = idxs_matrix[i, :]
                    X_in = X[idxs_y, :]
                    y_in = [y[idx] for idx in idxs_y]
                    auc_est = self.quick_auc_estimate(X_in, y_in)
                auc_estimates += [auc_est]
        sorted_indices = np.argsort(auc_estimates)[::-1]
        auc_estimates = [auc_estimates[i] for i in sorted_indices]
        idxs_matrix = idxs_matrix[sorted_indices]
        all_idxs_counts = dict((i, 0) for i in range(len(y)))
        accepted_rows = []
        for i in range(max_num_partitions):
            row_idx = self._suggest_row(idxs_matrix, accepted_rows, all_idxs_counts, min_seen_across_partitions)
            if row_idx is None:
                continue
            for x in idxs_matrix[row_idx, :]:
                all_idxs_counts[x] += 1
            accepted_rows += [row_idx]
        accepted_rows = [i for i in accepted_rows]
        random.shuffle(accepted_rows)
        idxs_matrix_ = np.zeros((len(accepted_rows), idxs_matrix.shape[1]), dtype=int)
        for i, row_idx in enumerate(accepted_rows):
            idxs_matrix_[i, :] = idxs_matrix[row_idx, :]
        idxs_matrix = idxs_matrix_[:]
        idxs_matrix = self.remove_redundancy_of_sampled_matrix(idxs_matrix)
        print(f"Indices matrix shape after redundancy removal: {idxs_matrix.shape}")
        self.idxs_matrix_report(idxs_matrix, y)
        for i in range(idxs_matrix.shape[0]):
            idxs = [int(x) for x in idxs_matrix[i, :]]
            yield idxs

    def idxs_matrix_report(self, idxs_matrix, y):
        print(f"Original positive negative balance: positive {np.sum(y)}, negative {len(y) - np.sum(y)}")
        n = idxs_matrix.shape[0]
        n_pos = 0
        n_neg = 0
        for i in range(n):
            idxs = [int(x) for x in idxs_matrix[i, :]]
            pos = np.sum([y[idx] for idx in idxs if idx < len(y) and idx >= 0])
            neg = len(idxs) - pos
            n_pos += pos
            n_neg += neg
        print(f"Avg positive samples: {n_pos/n}, avg negative samples: {n_neg/n}")