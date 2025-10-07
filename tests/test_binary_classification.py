import csv
import shutil

from lazyqsar.qsar import LazyBinaryQSAR
from sklearn.metrics import roc_auc_score

from lazyqsar.utils.logging import logger


def load_dataset(dataset_name):
    with open("benchmark/data/{0}_train.csv".format(dataset_name), "r") as f:
        reader = csv.reader(f)
        smiles_train = []
        y_train = []
        next(reader)
        for row in reader:
            smiles_train += [row[0]]
            y_train += [int(row[1])]
    logger.info("Number of training samples: {0}".format(len(y_train)))

    with open("benchmark/data/{0}_test.csv".format(dataset_name), "r") as f:
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
    smiles_train, y_train, smiles_test, y_test = load_dataset("ames")
    logger.info("Using featurizer")
    model = LazyBinaryQSAR(descriptor_type="chemeleon", mode="fast")
    model.fit(smiles_list=smiles_train, y=y_train)
    model.save("test_binary_classification")
    model = LazyBinaryQSAR.load("test_binary_classification")
    y_pred = model.predict_proba(smiles_list=smiles_test)[:, 1]
    logger.info("ROC-AUC: {0}".format(roc_auc_score(y_test, y_pred)))
    logger.info("Trying with ONNX")
    model.save_onnx("test_binary_classification")
    model = model.load_onnx("test_binary_classification")
    y_pred = model.predict_proba(smiles_list=smiles_test)[:, 1]
    logger.info("ROC-AUC: {0}".format(roc_auc_score(y_test, y_pred)))
    logger.info(
        "Removing temporary files from {0}".format("test_binary_classification")
    )
    shutil.rmtree("test_binary_classification")


if __name__ == "__main__":
    fit_and_evaluate()
