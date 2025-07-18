import os
import lazyqsar as lq
import tempfile
from sklearn.metrics import roc_curve, auc
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)

from tdc.single_pred import Tox

data = Tox(name="hERG")
split = data.get_split()
smiles_train = list(split["train"]["Drug"])
y_train = list(split["train"]["Y"])
smiles_valid = list(split["valid"]["Drug"])
y_valid = list(split["valid"]["Y"])


def fit():
    model = lq.LazyBinaryQSAR()
    model.fit(smiles_train, y_train)
    y_hat = model.predict(smiles_valid)
    fpr, tpr, _ = roc_curve(y_valid, y_hat)
    print("AUROC", auc(fpr, tpr))
    model.save_model(model_dir="herg_xg")


def predict():
    model_dir = os.path.abspath("herg_xg")
    print(f"Absolute path of the model directory: {model_dir}")
    model = lq.LazyBinaryQSAR.load_model(model_dir)
    y_hat = model.predict(smiles_valid)
    fpr, tpr, _ = roc_curve(y_valid, y_hat)
    print("AUROC", auc(fpr, tpr))


if __name__ == "__main__":
    predict()
    tmp = tempfile.TemporaryDirectory()
    tmp.cleanup()
