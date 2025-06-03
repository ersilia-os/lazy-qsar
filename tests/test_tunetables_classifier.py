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
split = data.get_split()
smiles_train = list(split["train"]["Drug"])
y_train = list(split["train"]["Y"])
smiles_valid = list(split["valid"]["Drug"])
y_valid = list(split["valid"]["Y"])

# NOTE: This version of tunetables supports arbitrary dataset size. This requires training.
# NOTE: If the dataset is small better to train them 1-4 epochs. 
# NOTE: For big (>15k) one 10 epoch works fine. It has default early stopping of 4 epoch,
# NOTE: I used internally in tunetables, a PCA method to reduce the dimensions of fetures of mordred to 100

def fit():
    model = lq.LazyBinaryQSAR(model_type="tunetables", descriptor_type="mordred")
    model.fit(smiles_train, y_train)
    model.save_model("bioavailability_ma")

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