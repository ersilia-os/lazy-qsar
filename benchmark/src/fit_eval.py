import os
import sys
import json
import pandas as pd
from sklearn.metrics import roc_curve, auc
from lazyqsar.qsar import LazyBinaryQSAR

from sklearn.metrics import roc_auc_score, average_precision_score

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

DATAPATH = os.path.join(root, "..", "data")
PREDSPATH = os.path.join(root, "..", "preds")

def evaluate_task(y_true, y_pred, metric):
    if metric == "roc-auc":
        return roc_auc_score(y_true, y_pred)
    elif metric == "pr-auc":
        return average_precision_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def train_eval(task, desc, metric):
    train = pd.read_csv(os.path.join(DATAPATH, f"{task}_train.csv"))
    test = pd.read_csv(os.path.join(DATAPATH, f"{task}_test.csv"))
    smiles_train = train["Drug"].tolist()
    y_train = train["Y"].tolist()
    smiles_test = test["Drug"].tolist()
    y_test = test["Y"].tolist()
    print(task, len(train), len(test))
    model = LazyBinaryQSAR(
                    descriptor_type=desc, 
                    mode="default"
                    )
    model.fit(smiles_train, y_train)
    y_pred_test = model.predict_proba(smiles_test)[:,1]
    test["pred"] = y_pred_test
    save_path = os.path.join(PREDSPATH, f"tdc_preds_{desc}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test.to_csv(
        os.path.join(save_path, "{}_test_pred.csv".format(task)), index=False
    )
    performance = evaluate_task(test["Y"], test["pred"], metric)
    return performance