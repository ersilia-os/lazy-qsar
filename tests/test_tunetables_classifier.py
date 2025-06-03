import os
import lazyqsar as lq
import tempfile
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
from tdc.single_pred import ADME

data = ADME(
    name='bioavailability_ma',
)
# from tdc.single_pred import Tox
# data = Tox(name='bioavailability_ma', path='../data/benchmark/bioavailability_ma')
split = data.get_split()
smiles_train = list(split["train"]["Drug"])
y_train = list(split["train"]["Y"])
smiles_valid = list(split["valid"]["Drug"])
y_valid = list(split["valid"]["Y"])

def fit():
    model = lq.LazyBinaryQSAR(model_type="zeroshot", descriptor_type="mordred")
    model.fit(smiles_train, y_train)
    y_hat = model.predict_proba(smiles_valid)[:,1]
    fpr, tpr, _ = roc_curve(y_valid, y_hat)
    print("AUROC", auc(fpr, tpr))

    # model.save_model(model_dir="bioavailability_ma")

def predict():
    model_dir = os.path.abspath("bioavailability_ma")
    print(f"Absolute path of the model directory: {model_dir}")
    model = lq.LazyBinaryQSAR.load_model(model_dir)
    y_hat = model.predict_proba(smiles_valid)[:,1]
    fpr, tpr, _ = roc_curve(y_valid, y_hat)
    print("AUROC", auc(fpr, tpr))


if __name__ == "__main__":
    fit()
    # predict()