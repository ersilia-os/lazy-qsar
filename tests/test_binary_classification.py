import csv
import os
import shutil
import random
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
        X_train = f["Values"][:]
    logger.info("Number of training samples: {0}".format(len(X_train)))

    with h5py.File(
        os.path.join(root, "data/{0}_test.h5".format(dataset_name)), "r"
    ) as f:
        X_test = f["Values"][:]
    logger.info("Number of testing samples: {0}".format(len(X_test)))

    _, y_train, _, y_test = load_dataset(dataset_name)

    return X_train, y_train, X_test, y_test


def fit_and_evaluate(mode="fast", clean=False, onnx=True, zip=True):
    if zip:
        output_dir_ = output_dir + ".zip"
    else:
        output_dir_ = output_dir
    logger.info("Binary classification task")
    smiles_train, y_train, smiles_test, y_test = load_dataset("bioavailability_ma")
    logger.info("Using featurizer")
    model = LazyBinaryQSAR(mode=mode)
    model.fit(smiles_list=smiles_train, y=y_train)
    model.save(output_dir_, onnx=onnx)
    model = LazyBinaryQSAR.load(output_dir_)
    y_pred = model.predict_proba(smiles_list=smiles_test)[:, 1]
    y_pred_train = model.predict_proba(smiles_list=smiles_train)[:, 1]
    logger.info("ROC-AUC train: {0}".format(roc_auc_score(y_train, y_pred_train)))
    logger.info(
        "Y pred train samples: {0}".format(random.sample(list(y_pred_train), 10))
    )

    logger.info("ROC-AUC: {0}".format(roc_auc_score(y_test, y_pred)))
    logger.info("Y pred samples: {0}".format(random.sample(list(y_pred), 10)))
    y_pred_train = model.predict_proba(smiles_list=smiles_train)[:, 1]
    if clean:
        logger.info("Removing temporary files from {0}".format(output_dir_))
        shutil.rmtree(output_dir_)


def fit_and_evaluate_agnostic(mode="fast", clean=False, onnx=True, zip=True):
    if zip:
        output_dir_ = output_dir + ".zip"
    else:
        output_dir_ = output_dir
    logger.info("Binary classification task")
    X_train, y_train, X_test, y_test = load_h5_dataset("bioavailability_ma")
    logger.info("Using agnostic model")
    model = LazyBinaryClassifier(mode=mode)
    model.fit(X=X_train, y=y_train)
    model.save(output_dir_, onnx=onnx)
    print("Saved model to {0}".format(output_dir))
    model = LazyBinaryClassifier.load(output_dir_)
    y_pred = model.predict_proba(X=X_test)[:, 1]
    logger.info("ROC-AUC: {0}".format(roc_auc_score(y_test, y_pred)))
    logger.info("Y-test pred samples: {0}".format(random.sample(list(y_pred), 10)))
    if clean:
        logger.info("Removing temporary files from {0}".format(output_dir_))
        shutil.rmtree(output_dir_)


if __name__ == "__main__":
    import argparse

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

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

    parser.add_argument(
        "--no-onnx", action="store_true", default=False, help="Do not use ONNX storing"
    )

    parser.add_argument(
        "--no-zip",
        action="store_true",
        default=False,
        help="Do not compress the output folder",
    )

    args = parser.parse_args()

    if args.agnostic:
        fit_and_evaluate_agnostic(
            mode=args.mode, clean=args.clean, onnx=not args.no_onnx, zip=not args.no_zip
        )
    else:
        fit_and_evaluate(
            mode=args.mode, clean=args.clean, onnx=not args.no_onnx, zip=not args.no_zip
        )
