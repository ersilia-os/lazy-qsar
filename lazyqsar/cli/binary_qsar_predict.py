import argparse
import csv
import os

import numpy as np
import pandas as pd

from ..agnostic import LazyBinaryClassifier
from ..qsar import DESCRIPTOR_TYPES


def read_smiles(input_csv):
    smiles_list = []
    with open(input_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            smiles_list += [r[0]]
    return smiles_list


def get_task_names(model_dir):
    task_names = []
    for dn in os.listdir(model_dir):
        if os.path.isdir(os.path.join(model_dir, dn)):
            task_names.append(dn)
    return sorted(task_names)


def get_featurizer_names(model_dir, tasks):
    featurizers_names = []
    for task_name in tasks:
        for dn in os.listdir(os.path.join(model_dir, task_name)):
            if os.path.isdir(os.path.join(model_dir, task_name, dn)):
                featurizers_names += [dn]
    featurizers_names = sorted(set(featurizers_names))
    return featurizers_names


def load_featurizer(model_dir, featurizer_name):
    featurizer = None
    for task_name in get_task_names(model_dir):
        model_subdir = os.path.join(model_dir, task_name, featurizer_name)
        if os.path.isdir(model_subdir):
            featurizer = DESCRIPTOR_TYPES[featurizer_name].load(model_subdir)
            break
    return featurizer


def main():
    parser = argparse.ArgumentParser(description="Predict with a LazyBinaryQSAR model.")

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the fitted model.",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV file containing the SMILES strings to predict on.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output file to save the predictions.",
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    input_csv = os.path.abspath(args.input_csv)
    output_csv = os.path.abspath(args.output_csv)

    smiles_list = read_smiles(input_csv)
    tasks = get_task_names(model_dir)
    featurizers = get_featurizer_names(model_dir, tasks)

    results = {}
    for featurizer_name in featurizers:
        featurizer = load_featurizer(model_dir, featurizer_name)
        X = featurizer.transform(smiles_list)
        for task_name in tasks:
            model_subdir = os.path.join(model_dir, task_name, featurizer_name)
            if os.path.isdir(model_subdir):
                model = LazyBinaryClassifier.load(model_subdir)
                y_pred = model.predict_proba(X)[:, 1]
                results[(task_name, featurizer_name)] = y_pred

    aggregated_results = {}
    for task_name in tasks:
        R = []
        for k, v in results.items():
            if k[0] == task_name:
                R += [v]
        aggregated_results[task_name] = np.average(np.array(R), axis=0)

    R = []
    for task in tasks:
        R += [aggregated_results[task]]
    R = np.array(R).T

    df = pd.DataFrame(R, columns=tasks)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
