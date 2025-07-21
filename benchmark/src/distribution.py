import numpy as np
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt

FIGUREPATH = "../figures"
DATAPATH = "../data"
PREDSPATH = "../predictions"

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

from defaults import ADMET_CLF_TASKS

# Compare METHODS

descs = ["chemeleon", "morgan"]
models = ["logistic_regression", "random_forest", "tune_tables"]

for model in models:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for j, desc in enumerate(descs):
        ax = axes[j]
        predictions = {}

        for assay in ADMET_CLF_TASKS:
            try:
                file_path = os.path.join(
                    PREDSPATH, f"tdc_preds_{model}_{desc}", f"{assay}_test_pred.csv"
                )
                df = pd.read_csv(file_path)
                predictions[assay] = df["pred"].tolist()
            except FileNotFoundError:
                print(f"Missing: {assay} for {model}-{desc}")
                continue

        if not predictions:
            ax.set_title(f"{desc} (No data)")
            continue

        x_labels = list(predictions.keys())
        x_pos = np.arange(len(x_labels))

        for i, assay in enumerate(x_labels):
            preds = predictions[assay]
            jitter = np.random.normal(loc=0, scale=0.05, size=len(preds))
            ax.scatter(np.full(len(preds), x_pos[i]) + jitter, preds, alpha=0.6, s=10)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_title(f"{desc}")
        ax.set_xlabel("Assays")
        if j == 0:
            ax.set_ylabel("")

    fig.suptitle(f"Distribution for {model}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the combined figure
    outname = f"distribution_{model}.png"
    plt.savefig(os.path.join(FIGUREPATH, outname), dpi=300)
    plt.close()
