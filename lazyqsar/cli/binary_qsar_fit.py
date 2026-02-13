import os
import csv
import shutil
import argparse

import numpy as np

from ..agnostic import LazyBinaryClassifier
from ..qsar import DESCRIPTOR_TYPES, DESCRIPTORS_MODE


def read_all_smiles(data_dir):
    smiles_list = []
    for fn in os.listdir(data_dir):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(data_dir, fn), "r") as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                smiles_list += [r[0]]
    smiles_list = list(set(smiles_list))
    return smiles_list


def get_task_names(data_dir):
    task_names = []
    for fn in os.listdir(data_dir):
        if not fn.endswith(".csv"):
            continue
        task_names += [os.path.splitext(fn)[0]]
    return task_names


def get_task_data(data_dir, task_name):
    smiles_list = []
    y = []
    with open(os.path.join(data_dir, task_name + ".csv"), "r") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            smiles_list += [r[0]]
            y += [int(r[1])]
    return smiles_list, np.array(y, dtype=int)


def main():

    parser = argparse.ArgumentParser(description="Fit a LazyBinaryQSAR model.")
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        help="Mode for the LazyBinaryQSAR (fast, default or slow).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the training data.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory to save the fitted model.",
    )
    args = parser.parse_args()

    if os.path.exists(args.model_dir):
        raise FileExistsError(
            f"Model directory {args.model_dir} already exists. Please remove it before running this command."
        )

    task_names = get_task_names(args.data_dir)
    descriptor_types = DESCRIPTORS_MODE[args.mode]

    all_smiles = read_all_smiles(args.data_dir)
    all_smiles2idx = {s: i for i, s in enumerate(all_smiles)}

    for descriptor_type in descriptor_types:
        if descriptor_type not in DESCRIPTOR_TYPES:
            raise Exception(f"Descriptor type {descriptor_type} is not supported.")
        descriptor = DESCRIPTOR_TYPES[descriptor_type]()
        X = descriptor.transform(all_smiles)
        for task_name in task_names:
            model_subdir = os.path.join(args.model_dir, task_name, descriptor_type)
            if not os.path.exists(model_subdir):
                os.makedirs(model_subdir)
            descriptor.save(model_subdir)
            shutil.copy(
                os.path.join(model_subdir, "featurizer.json"),
                os.path.join(args.model_dir, f"{descriptor_type}.json"),
            )
        np.save(os.path.join(args.model_dir, f"{descriptor_type}.npy"), X)

    data = {}
    for task_name in task_names:
        smiles_list, y = get_task_data(args.data_dir, task_name)
        data[task_name] = (smiles_list, y)

    for descriptor_type in descriptor_types:
        X = np.load(os.path.join(args.model_dir, f"{descriptor_type}.npy"))
        for task_name in task_names:
            idxs = [all_smiles2idx[s] for s in data[task_name][0]]
            y = data[task_name][1]
            X_task = X[idxs]
            model = LazyBinaryClassifier(mode=args.mode)
            model.fit(X=X_task, y=y)
            model_subdir = os.path.join(args.model_dir, task_name, descriptor_type)
            model.save(model_subdir)
            shutil.copy(
                os.path.join(args.model_dir, f"{descriptor_type}.json"),
                os.path.join(model_subdir, "featurizer.json"),
            )
        os.remove(os.path.join(args.model_dir, f"{descriptor_type}.json"))
        os.remove(os.path.join(args.model_dir, f"{descriptor_type}.npy"))


if __name__ == "__main__":
    main()
