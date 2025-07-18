import lazyqsar
import os
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.model_selection import train_test_split


file_path = "data/bioavailability_ma.tab"

df = pd.read_csv(file_path, sep="\t")
df = df.dropna(subset=["Drug", "Y"])
df = df[df["Y"].isin([0, 1])]


smiles_list = list(df["Drug"])
y_list = list(df["Y"])

train_idxs, test_idxs = train_test_split(
    range(len(smiles_list)), test_size=0.2, random_state=42, stratify=y_list
)

smiles_train = [smiles_list[i] for i in train_idxs]
y_train = [y_list[i] for i in train_idxs]
smiles_valid = [smiles_list[i] for i in test_idxs]
y_valid = [y_list[i] for i in test_idxs]


model_type = "logistic_regression"


def fit():
    import time

    st = time.perf_counter()
    model = lazyqsar.LazyBinaryQSAR(descriptor_type="chemeleon", model_type=model_type)
    model.fit(smiles_train, y_train)
    model_path = os.path.abspath(model_type)
    model.save_model(model_path)
    y_hat = model.predict_proba(smiles_valid)
    fpr, tpr, _ = roc_curve(y_valid, y_hat)
    print("AUROC", auc(fpr, tpr))
    et = time.perf_counter()
    print(f"Training takes: {et - st:.4} seconds")


def predict():
    import time

    print(f"Length of the X sample: {len(smiles_valid)}")
    st = time.perf_counter()
    model_path = os.path.abspath(model_type)
    model = lazyqsar.LazyBinaryQSAR.load_model(model_path)
    y_hat = model.predict_proba(smiles_valid)
    fpr, tpr, _ = roc_curve(y_valid, y_hat)
    print("##########################################")
    print("AUROC", auc(fpr, tpr))
    et = time.perf_counter()
    print(f"Training takes: {et - st:.4} seconds")
    print("##########################################")


if __name__ == "__main__":
    fit()
    # predict()
