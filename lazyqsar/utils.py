import random
import math
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

class SamplingUtils(object):

    def __init__(self):
        pass

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

    def get_partition_indices(self, y,
                            min_positive_proportion=0.01,
                            max_positive_proportion=0.5,
                            min_samples=30,
                            max_samples=10000,
                            min_positive_samples=10,
                            max_num_partitions=100):
        pos_idxs = [i for i, y_ in enumerate(y) if y_ == 1]
        neg_idxs = [i for i, y_ in enumerate(y) if y_ == 0]
        random.shuffle(pos_idxs)
        random.shuffle(neg_idxs)
        n_tot = len(y)
        n_pos = len(pos_idxs)
        n_neg = len(neg_idxs)
        print(f"Total samples: {n_tot}, positive samples: {n_pos}, negative samples: {n_neg}")
        print(f"Positive proportion: {n_pos / n_tot:.2f}")
        if n_tot < min_samples:
            raise Exception("Not enough samples.")
        n_samples = min(n_tot, max_samples)
        real_pos_prop = n_pos / n_tot
        upper_pos_prop = min(n_pos / n_samples, max_positive_proportion)
        if real_pos_prop <= min_positive_proportion:
            if upper_pos_prop <= min_positive_proportion:
                desired_pos_prop = min_positive_proportion
            else:
                desired_pos_prop = upper_pos_prop
        elif real_pos_prop >= max_positive_proportion:
            desired_pos_prop = max_positive_proportion
        else:
            desired_pos_prop = upper_pos_prop
        n_pos_samples = min(int(n_samples * desired_pos_prop) + 1, n_pos)
        n_neg_samples = min(n_neg, int(n_pos_samples * (1 - desired_pos_prop) / desired_pos_prop) + 1, n_samples - n_pos_samples)
        n_samples = n_pos_samples + n_neg_samples
        if n_pos_samples < min_positive_samples:
            raise Exception("Not enough positive samples.")
        if n_neg_samples < min_positive_samples:
            raise Exception("Not enough negative samples.")
        pos_sampling_rounds = min(self.min_sampling_rounds(n_pos, n_pos_samples), max_num_partitions)
        neg_sampling_rounds = min(self.min_sampling_rounds(n_neg, n_neg_samples), max_num_partitions)
        sampling_rounds = max(pos_sampling_rounds, neg_sampling_rounds)
        print(f"Sampling rounds: {sampling_rounds}, positive samples per round: {n_pos_samples}, negative samples per round: {n_neg_samples}")
        print(f"Desired positive proportion: {desired_pos_prop}", "Actual positive proportion: ", n_pos_samples / n_samples)
        for _ in range(sampling_rounds):
            pos_idxs_sampled = random.sample(pos_idxs, n_pos_samples)
            neg_idxs_sampled = random.sample(neg_idxs, n_neg_samples)
            sampled_idxs = pos_idxs_sampled + neg_idxs_sampled
            random.shuffle(sampled_idxs)
            yield sampled_idxs