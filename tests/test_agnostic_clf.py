import lazyqsar
import os
import time
import sys
from sklearn.metrics import roc_curve, auc
import pandas as pd

model_type = sys.argv[1]

train = pd.read_csv("../benchmark/data/bioavailability_ma_train.csv")
y_train = train["Y"].tolist()
test = pd.read_csv("../benchmark/data/bioavailability_ma_test.csv")
y_test = test["Y"].tolist()

# Precalculated descriptors for train and test sets
X_train = "../benchmark/data/ersilia_preds/bioavailability_ma_train_eos9o72.h5"
X_test = "../benchmark/data/ersilia_preds/bioavailability_ma_test_eos9o72.h5"


def fit():
    st = time.perf_counter()
    model = lazyqsar.LazyBinaryClassifier(model_type=model_type)
    model.fit(h5_file=X_train, y=y_train)
    model_path = os.path.abspath("test_model")
    model.save_model(model_path)
    et = time.perf_counter()
    print(f"Training takes: {et - st:.4} seconds")


def predict():
    st = time.perf_counter()
    model_path = os.path.abspath("test_model")
    model = lazyqsar.LazyBinaryClassifier.load_model(model_path)
    y_hat = model.predict_proba(h5_file=X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_hat)
    print("AUROC", auc(fpr, tpr))
    et = time.perf_counter()
    print(f"Predicting takes: {et - st:.4} seconds")


if __name__ == "__main__":
    fit()
    predict()
