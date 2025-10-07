import csv

from lazyqsar.descriptors.descriptors import MorganFingerprint
from lazyqsar.utils.logging import logger

prefix = "ames"
descriptor_class = MorganFingerprint

with open("benchmark/data/{0}_train.csv".format(prefix), "r") as f:
    reader = csv.reader(f)
    smiles_train = []
    y_train = []
    next(reader)
    for row in reader:
        smiles_train += [row[0]]
        y_train += [int(row[1])]
logger.info("Number of training samples: {0}".format(len(y_train)))
descriptor = descriptor_class()
descriptor.fit(smiles_train)
X_train = descriptor.transform(smiles_train)
logger.info("Shape of training descriptors: {0}".format(X_train.shape))

with open("benchmark/data/{0}_test.csv".format(prefix), "r") as f:
    reader = csv.reader(f)
    smiles_test = []
    y_test = []
    next(reader)
    for row in reader:
        smiles_test += [row[0]]
        y_test += [int(row[1])]
print("Number of testing samples: ", len(y_test))
descriptor = descriptor_class()
descriptor.fit(smiles_test)
X_test = descriptor.transform(smiles_test)
print("Shape of testing descriptors: ", X_test.shape)

import numpy as np

X = X_train
y = np.array(y_train)

from lazyqsar.latent_variables.old.sparse import get_reducer_parameters
from lazyqsar.latent_variables.old.sparse import SparseDimReducerBinaryClassification

red_params = get_reducer_parameters(X, y)

reducer = SparseDimReducerBinaryClassification(
    k_features=red_params["k_features"],
    n_components=red_params["n_components"],
    random_state=42,
)
reducer.fit(X, y)
X_train = reducer.transform(X_train, y_train)
X_test = reducer.transform(X_test)
print("Shape of reduced training descriptors: ", X_train.shape)
print("Shape of reduced testing descriptors: ", X_test.shape)

from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(
    class_weight="balanced",
    cv=5,
    random_state=42,
    scoring="roc_auc",
    max_iter=1000,
    n_jobs=-1,
)

from sklearn.svm import LinearSVC

clf = LinearSVC()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
print(y_proba)

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

print("ROC-AUC: ", roc_auc_score(y_test, y_proba))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1-score: ", f1_score(y_test, y_pred))
