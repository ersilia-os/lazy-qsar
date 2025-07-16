import lazyqsar
import numpy as np
import warnings
import os
from sklearn.metrics import roc_curve, auc


warnings.filterwarnings("ignore", category=ResourceWarning)
from tdc.single_pred import ADME

data = ADME(
    name='bioavailability_ma',
)
split = data.get_split()
smiles_train = list(split["train"]["Drug"])
y_train = list(split["train"]["Y"])
smiles_valid = list(split["valid"]["Drug"])
y_valid = list(split["valid"]["Y"])


def fit():
    import time
    st = time.perf_counter()
    model = lazyqsar.LazyBinaryQSAR(descriptor_type="chemeleon", model_type="tunetables")
    model.fit(smiles_train, y_train)
    model_path = os.path.abspath("tunetables_chemeleon_new_4_epoch")
    model.save_model(model_path)
    y_hat = model.predict_proba(smiles_valid)
    fpr, tpr, _ = roc_curve(y_valid, y_hat)
    print("AUROC", auc(fpr, tpr))
    et = time.perf_counter()
    print(f"Training takes: {et-st:.4} seconds")

def predict():
    import time
    print(f"Length of the X sample: {len(smiles_valid)}")
    st = time.perf_counter()
    model_path = os.path.abspath("tunetables_chemetunetables_chemeleon_new_4_epochleon_new")
    model = lazyqsar.LazyBinaryQSAR.load_model(model_path)
    y_hat = model.predict_proba(smiles_valid)
    fpr, tpr, _ = roc_curve(y_valid, y_hat)
    print("AUROC", auc(fpr, tpr))
    et = time.perf_counter()
    print(f"Training takes: {et-st:.4} seconds")

if __name__ == "__main__":
    predict()

