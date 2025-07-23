import lazyqsar
import os
import time
import sys
from sklearn.metrics import roc_curve, auc
import pandas as pd

model_type = sys.argv[1]
desc = sys.argv[2]

train = pd.read_csv("../benchmark/data/bioavailability_ma_train.csv")
smiles_train = train["Drug"].tolist()
y_train = train["Y"].tolist()
test = pd.read_csv("../benchmark/data/bioavailability_ma_test.csv")
smiles_test = test["Drug"].tolist()
y_test = test["Y"].tolist()

print(len(smiles_train))


def fit():
    st = time.perf_counter()
    model = lazyqsar.LazyBinaryQSAR(descriptor_type=desc, model_type=model_type)
    model.fit(smiles_train, y_train)
    model_path = os.path.abspath("test_model")
    model.save_model(model_path)
    et = time.perf_counter()
    print(f"Training takes: {et - st:.4} seconds")


def predict():
    print(f"Length of the X sample: {len(smiles_test)}")
    st = time.perf_counter()
    model_path = os.path.abspath("test_model")
    model = lazyqsar.LazyBinaryQSAR.load_model(model_path)
    y_hat = model.predict_proba(smiles_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_hat)
    print("AUROC", auc(fpr, tpr))
    et = time.perf_counter()
    print(f"Predicting takes: {et - st:.4} seconds")


if __name__ == "__main__":
    fit()
    predict()
