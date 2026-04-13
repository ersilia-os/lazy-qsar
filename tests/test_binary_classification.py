import csv
import os
import shutil
import random
import time
import h5py
import argparse

from lazyqsar.qsar import LazyClassifierQSAR
from lazyqsar.agnostic import LazyClassifier
from lazyqsar.descriptors.morgan import MorganFingerprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from lazyqsar.utils.logging import logger


def rf_baseline_score(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    return auc, elapsed


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


def fit_and_evaluate(mode="default", clean=False, onnx=True, zip=True):
    if zip:
        output_dir_ = output_dir + ".zip"
    else:
        output_dir_ = output_dir
    logger.info("Binary classification task")
    smiles_train, y_train, smiles_test, y_test = load_dataset("bioavailability_ma")

    logger.info("Computing RF baseline (Morgan fingerprints)...")
    morgan = MorganFingerprint()
    X_train_morgan = morgan.transform(smiles_train)
    X_test_morgan = morgan.transform(smiles_test)
    rf_auc, rf_time = rf_baseline_score(X_train_morgan, y_train, X_test_morgan, y_test)
    logger.info("RF baseline ROC-AUC: {0:.4f} (train time: {1:.1f}s)".format(rf_auc, rf_time))

    logger.info("Using featurizer")
    model = LazyClassifierQSAR(mode=mode)
    t0 = time.time()
    model.fit(smiles_list=smiles_train, y=y_train)
    lazy_time = time.time() - t0
    model.save(output_dir_, onnx=onnx)
    model = LazyClassifierQSAR.load(output_dir_)
    y_pred = model.predict_proba(smiles_list=smiles_test)[:, 1]
    lazy_auc = roc_auc_score(y_test, y_pred)

    logger.info("ROC-AUC: {0:.4f} (LazyClassifierQSAR, {1:.1f}s) vs {2:.4f} (RF, {3:.1f}s)".format(
        lazy_auc, lazy_time, rf_auc, rf_time))
    logger.info("Y pred samples: {0}".format(random.sample(list(y_pred), 10)))
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

    logger.info("Computing RF baseline...")
    rf_auc, rf_time = rf_baseline_score(X_train, y_train, X_test, y_test)
    logger.info("RF baseline ROC-AUC: {0:.4f} (train time: {1:.1f}s)".format(rf_auc, rf_time))

    logger.info("Using agnostic model")
    model = LazyClassifier()
    t0 = time.time()
    model.fit(X=X_train, y=y_train)
    lazy_time = time.time() - t0

    # LR vs XGB prediction correlation (before save/load)
    base = model.model.models[0]
    X_test_prep = base.prep.transform(X_test)
    lr_pred = base.lr.predict_proba(X_test_prep)[:, 1]
    xgb_pred = base.xgb.predict_proba(X_test_prep)[:, 1]

    model.save(output_dir_, onnx=onnx)
    model = LazyClassifier.load(output_dir_)
    y_pred = model.predict_proba(X=X_test)[:, 1]
    lazy_auc = roc_auc_score(y_test, y_pred)

    print("ROC-AUC: {0:.4f} (LazyClassifierQSAR, {1:.1f}s) vs {2:.4f} (RF, {3:.1f}s)".format(
        lazy_auc, lazy_time, rf_auc, rf_time), flush=True)
    n = len(lr_pred)
    lr_mean = sum(lr_pred) / n
    xgb_mean = sum(xgb_pred) / n
    cov = sum((lr_pred[i] - lr_mean) * (xgb_pred[i] - xgb_mean) for i in range(n)) / n
    lr_std = (sum((lr_pred[i] - lr_mean) ** 2 for i in range(n)) / n) ** 0.5
    xgb_std = (sum((xgb_pred[i] - xgb_mean) ** 2 for i in range(n)) / n) ** 0.5
    corr = cov / (lr_std * xgb_std) if lr_std * xgb_std > 0 else float("nan")

    bins = 10
    width = 40
    print("\nLR vs XGB predicted probabilities (Pearson r={:.3f}):".format(corr))
    print("  {:<5} | {}".format("LR\\XGB", " " * width))
    edges = [i / bins for i in range(bins + 1)]
    header_labels = "".join("{:<4}".format("{:.1f}".format(edges[i])) for i in range(0, bins + 1, 2))
    print("  {:<5} | 0{}1".format("", " " * (width - 2)))
    grid = [[0] * bins for _ in range(bins)]
    for i in range(len(lr_pred)):
        xi = min(int(xgb_pred[i] * bins), bins - 1)
        yi = min(int(lr_pred[i] * bins), bins - 1)
        grid[yi][xi] += 1
    max_count = max(c for row in grid for c in row) or 1
    chars = " .:-=+*#@"
    for row_i in range(bins - 1, -1, -1):
        label = "{:.1f}".format(edges[row_i])
        cells = "".join(chars[min(int(c / max_count * (len(chars) - 1)), len(chars) - 1)] for c in grid[row_i])
        print("  {:<5} |{}|".format(label, cells))
    print("  {:<5} +{}+".format("", "-" * bins))
    print("  {:<5}  {}".format("", "0" + " " * (bins - 2) + "1"))
    print("         XGB predicted probability\n")

    if clean:
        logger.info("Removing temporary files from {0}".format(output_dir_))
        shutil.rmtree(output_dir_)


if __name__ == "__main__":
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
