import csv
import shutil

from lazyqsar.qsar import LazyBinaryQSAR
from lazyqsar.agnostic import LazyBinaryClassifier
from sklearn.metrics import roc_auc_score

from lazyqsar.utils.logging import logger


def load_dataset(dataset_name):
    with open("data/{0}_train.csv".format(dataset_name), "r") as f:
        reader = csv.reader(f)
        smiles_train = []
        y_train = []
        next(reader)
        for row in reader:
            smiles_train += [row[0]]
            y_train += [int(row[1])]
    logger.info("Number of training samples: {0}".format(len(y_train)))

    with open("data/{0}_test.csv".format(dataset_name), "r") as f:
        reader = csv.reader(f)
        smiles_test = []
        y_test = []
        next(reader)
        for row in reader:
            smiles_test += [row[0]]
            y_test += [int(row[1])]
    logger.info("Number of testing samples: {0}".format(len(y_test)))
    return smiles_train, y_train, smiles_test, y_test


def fit_and_evaluate():
    logger.info("Binary classification task")
    smiles_train, y_train, smiles_test, y_test = load_dataset("bioavailability_ma")
    logger.info("Using featurizer")
    model = LazyBinaryQSAR(descriptor_type="chemeleon", mode="fast")
    model.fit(smiles_list=smiles_train, y=y_train)
    model.save("test_binary_classification")
    model = LazyBinaryQSAR.load("test_binary_classification")
    y_pred = model.predict_proba(smiles_list=smiles_test)[:, 1]
    logger.info("ROC-AUC: {0}".format(roc_auc_score(y_test, y_pred)))
    logger.info(
        "Removing temporary files from {0}".format("test_binary_classification")
    )
    shutil.rmtree("test_binary_classification")


def load_descriptors(
    dataset_name,
    ):  # precalculated using Ersilia by each user individually
    X_train = f"data/{dataset_name}_train.h5"
    X_test = f"data/{dataset_name}_test.h5"
    return X_train, X_test


def fit_and_evaluate_agnostic():
    model = LazyBinaryClassifier()
    smiles_train, y_train, smiles_test, y_test = load_dataset("bioavailability_ma")
    X_train, X_test = load_descriptors("bioavailability_ma")
    model.fit(h5_file=X_train, y=y_train)
    model.save("test_binary_agnostic_classification")
    model = LazyBinaryClassifier.load("test_binary_agnostic_classification")
    y_pred = model.predict_proba(h5_file=X_test)[:, 1]
    logger.info("ROC-AUC: {0}".format(roc_auc_score(y_test, y_pred)))
    logger.info(
        "Removing temporary files from {0}".format(
            "test_binary_agnostic_classification"
        )
    )
    shutil.rmtree("test_binary_agnostic_classification")


if __name__ == "__main__":
    fit_and_evaluate()
    #fit_and_evaluate_agnostic()
