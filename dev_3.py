from lazyqsar.assemblers.eclectic_binary_classifier import convert_to_onnx

convert_to_onnx("test_dev_2")

import csv

from lazyqsar.descriptors.descriptors import ChemeleonDescriptor, MorganFingerprint
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

from lazyqsar.artifacts.artifact_binary_classifier import LazyBinaryClassifierArtifact

model = LazyBinaryClassifierArtifact.load("test_dev_2")

y_pred = model.predict(X_test)
print(y_pred)