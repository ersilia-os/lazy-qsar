import csv
import os
import shutil
import h5py

from lazyqsar.qsar import LazyBinaryQSAR
from lazyqsar.agnostic import LazyBinaryClassifier
from sklearn.metrics import roc_auc_score

from lazyqsar.utils.logging import logger


root = os.path.dirname(os.path.abspath(__file__))

output_dir = "lazyqsar_test_output"


def load_dataset(dataset_name):
    with open(os.path.join(root, "data/{0}_train.csv".format(dataset_name)), "r") as f:
        reader = csv.reader(f)
        smiles_train = []
        y_train = []
        next(reader)
        for row in reader:
            smiles_train += [row[0]]
            y_train += [int(row[1])]
    logger.info("Number of training samples: {0}".format(len(y_train)))

    with open(os.path.join(root, "data/{0}_test.csv".format(dataset_name)), "r") as f:
        reader = csv.reader(f)
        smiles_test = []
        y_test = []
        next(reader)
        for row in reader:
            smiles_test += [row[0]]
            y_test += [int(row[1])]
    logger.info("Number of testing samples: {0}".format(len(y_test)))
    return smiles_train, y_train, smiles_test, y_test


def load_h5_dataset(dataset_name):
    with h5py.File(
        os.path.join(root, "data/{0}_train.h5".format(dataset_name)), "r"
    ) as f:
        X_train = f["X"][:]
        y_train = f["y"][:]
    logger.info("Number of training samples: {0}".format(len(y_train)))

    with h5py.File(
        os.path.join(root, "data/{0}_test.h5".format(dataset_name)), "r"
    ) as f:
        X_test = f["X"][:]
        y_test = f["y"][:]
    logger.info("Number of testing samples: {0}".format(len(y_test)))
    return X_train, y_train, X_test, y_test


def fit_and_evaluate(mode="fast", clean=False):
    logger.info("Binary classification task")
    smiles_train, y_train, smiles_test, y_test = load_dataset("bioavailability_ma")
    logger.info("Using featurizer")
    model = LazyBinaryQSAR(mode=mode)
    model.fit(smiles_list=smiles_train, y=y_train)
    model.save(output_dir)
    model = LazyBinaryQSAR.load(output_dir)
    y_pred = model.predict_proba(smiles_list=smiles_test)[:, 1]
    logger.info("ROC-AUC: {0}".format(roc_auc_score(y_test, y_pred)))
    if clean:
        logger.info(
            "Removing temporary files from {0}".format(output_dir)
        )
        shutil.rmtree(output_dir)


def fit_and_evaluate_agnostic(mode="fast", clean=False):
    logger.info("Binary classification task")
    X_train, y_train, X_test, y_test = load_h5_dataset("bioavailability_ma")
    logger.info("Using agnostic model")
    model = LazyBinaryClassifier(mode=mode)
    model.fit(X=X_train, y=y_train)
    model.save(output_dir)
    model = LazyBinaryClassifier.load(output_dir)
    y_pred = model.predict_proba(X=X_test)[:, 1]
    logger.info("ROC-AUC: {0}".format(roc_auc_score(y_test, y_pred)))
    if clean:
        logger.info(
            "Removing temporary files from {0}".format(output_dir)
        )
        shutil.rmtree(output_dir)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        help="Mode of operation: fast, default or slow",
    )

    parser.add_argument(
        "--agnostic",
        action="store_true",
        help="Use agnostic model for binary classification",
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Remove temporary files after evaluation",
    )
    args = parser.parse_args()

    if args.agnostic:
        fit_and_evaluate_agnostic(mode=args.mode, clean=args.clean)
    else:
        fit_and_evaluate(mode=args.mode, clean=args.clean)