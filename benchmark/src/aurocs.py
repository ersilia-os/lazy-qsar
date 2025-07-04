import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import stylia as st
from stylia import TWO_COLUMNS_WIDTH, ONE_COLUMN_WIDTH
from sklearn.metrics import roc_curve, auc


FIGUREPATH =  "../figures"
DATAPATH = "../data"

# Compare METHODS

desc = sys.argv[1]
"""
models = ["xgboost", "xgboost_pca","tunetables", "zsrandomforest"]

clf_datasets = ["bioavailability_ma", "hia_hou", "pgp_broccatelli", "bbb_martins", "cyp2c9_veith","cyp2d6_veith",
                  "cyp3a4_veith", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
                  "cyp3a4_substrate_carbonmangels","herg","ames", "dili"]

c = {
    "xgboost": "#dca0dc",
    "xgboost_pca": "#fad782",
    "tunetables": "#faa08c",
    "zsrandomforest": "#aa96fa"}
"""
models = ["xgboost", "xgboost_pca", "zsrandomforest", "randomforest"]

clf_datasets = ["cyp2c9_veith","cyp2d6_veith",
                "cyp3a4_veith", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
                "dili"]

c = {
    "xgboost": "#dca0dc",
    "xgboost_pca": "#fad782",
    "randomforest": "#faa08c",
    "zsrandomforest": "#aa96fa"}

fig, axs = st.create_figure(3,2, width=ONE_COLUMN_WIDTH, height =TWO_COLUMNS_WIDTH)
for i,d in enumerate(clf_datasets):
    results_dict = {}
    for model in models:
        results_dict[model] = os.path.join(DATAPATH, f"tdc_preds_{model}_{desc}", "{}_test_1.csv".format(d))
    ax = axs.next()
    for k,v in results_dict.items():
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
plt.savefig(os.path.join(FIGUREPATH, f"aurocs_{desc}_rf.png"), dpi=300)