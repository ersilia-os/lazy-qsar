import os
import sys
import json
import pandas as pd
from lazyqsar.qsar import LazyBinaryQSAR

from sklearn.metrics import roc_auc_score, average_precision_score

DATAPATH = "../data"
PREDSPATH = "../predictions"

model_type = sys.argv[1]
desc = sys.argv[2]

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

from defaults import ADMET_CLF_TASKS

for a in ADMET_CLF_TASKS.keys():
    train = pd.read_csv(f"../data/{a}_train.csv")
    smiles_train = train["Drug"].tolist()
    y_train = train["Y"].tolist()
    test = pd.read_csv(f"../data/{a}_test.csv")
    smiles_test = test["Drug"].tolist()
    y_test = test["Y"].tolist()
    print(a, len(train), len(test))
    model = LazyBinaryQSAR(model_type=model_type, descriptor_type=desc)
    model.fit(smiles_train, y_train)
    y_pred_test = model.predict_proba(smiles_test)[:,1]
    test["pred"] = y_pred_test
    save_path = os.path.join(PREDSPATH, f"tdc_preds_{model_type}_{desc}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test.to_csv(os.path.join(save_path, "{}_test_pred.csv".format(a)), index=False)


def evaluate_task(y_true, y_pred, metric):
    if metric == "roc-auc":
        return roc_auc_score(y_true, y_pred)
    elif metric == "pr-auc":
        return average_precision_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")


predictions = {}
for k, v in ADMET_CLF_TASKS.items():
    try:
        test = pd.read_csv(
            os.path.join(
                PREDSPATH,
                f"tdc_preds_{model_type}_{desc}",
                f"{k}_test_pred.csv",
            )
        )
    except FileNotFoundError:
        print(f"Skipping {k} as the file does not exist.")
        continue
    perf = evaluate_task(test["Y"], test["pred"], v["metric"])
    predictions[k] = perf

print(predictions)

results_file = f"{model_type}_{desc}.json"

with open(
    os.path.join(PREDSPATH, f"tdc_preds_{model_type}_{desc}", results_file), "w"
) as f:
    json.dump(predictions, f, indent=2)
