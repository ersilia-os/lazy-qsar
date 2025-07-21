import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import stylia as st
from stylia import TWO_COLUMNS_WIDTH, ONE_COLUMN_WIDTH
from sklearn.metrics import roc_curve, auc

FIGUREPATH = "../figures"
PREDSPATH = "../predictions"

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

from defaults import ADMET_CLF_TASKS

# Compare METHODS
descs = ["chemeleon", "morgan"]
models = ["logistic_regression", "random_forest", "tune_tables"]

c = {
    "logistic_regression": "#50285a",
    "random_forest": "#fad782",
    "tune_tables": "#aa96fa",
}

for desc in descs:
    fig, axs = st.create_figure(7, 2, width=ONE_COLUMN_WIDTH, height=TWO_COLUMNS_WIDTH)
    for i, d in enumerate(ADMET_CLF_TASKS.keys()):
        results_dict = {}
        for model in models:
            results_dict[model] = os.path.join(
                PREDSPATH, f"tdc_preds_{model}_{desc}", "{}_test_pred.csv".format(d)
            )
        ax = axs.next()
        for k, v in results_dict.items():
            try:
                test = pd.read_csv(v)
                fpr, tpr, _ = roc_curve(test["Y"], test["pred"])
                auc_score = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{k} (AUC={auc_score:.2f})", color=c[k])
                ax.set_title(d)
                ax.set_xlabel("")
                ax.set_ylabel("")
            except:
                print("file not found, skipping")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGUREPATH, f"aurocs_{desc}.png"), dpi=300)
