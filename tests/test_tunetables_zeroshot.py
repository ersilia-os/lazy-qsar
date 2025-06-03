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

# NOTE: Zero-shot only supports dataset size less or equals to 1000
# NOTE: You can batch them and aggregate the result as well
# NOTE: I used internally in tunetables, a PCA method to reduce the dimensions of fetures of mordred to 100
# NOTE: Each prediction run results a slight different results in accuracy 
# [this is inrentional, a random permutation in the label class yileds different results and with ensemble config, the results are aggregated]


def fit_predict():
    model = lq.LazyBinaryQSAR(model_type="zeroshot", descriptor_type="mordred")
    model.fit(smiles_train, y_train)
    y_hat = model.predict_proba(smiles_valid)[:,1]
    fpr, tpr, _ = roc_curve(y_valid, y_hat)
    print("AUROC", auc(fpr, tpr))

if __name__ == "__main__":
    fit_predict()