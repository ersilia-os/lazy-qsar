import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
import sys

FIGUREPATH =  "../figures"
DATAPATH = "../data"

# Compare METHODS
"""
descs = ["morgan", "mordred"]
models = ["zstunetables","tunetables","xgboost", "xgboost_pca", "zsrandomforest"]
assays = ["bioavailability_ma", "hia_hou", "pgp_broccatelli", "bbb_martins", "cyp2c9_veith","cyp2d6_veith",
                  "cyp3a4_veith", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
                  "cyp3a4_substrate_carbonmangels","herg","ames", "dili"]
"""
descs = ["morgan", "mordred"]
models = ["xgboost", "xgboost_pca", "zsrandomforest", "randomforest"]
assays = ["cyp2c9_veith","cyp2d6_veith",
                "cyp3a4_veith", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
                "dili"]

seed = 1

for model in models:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    for j, desc in enumerate(descs):
        ax = axes[j]
        predictions = {}

        for assay in assays:
            try:
                file_path = os.path.join(
                    DATAPATH,
                    f"tdc_preds_{model}_{desc}",
                    f"{assay}_test_{seed}.csv"
                )
                df = pd.read_csv(file_path)
                predictions[assay] = df["pred"].tolist()
            except FileNotFoundError:
                print(f"Missing: {assay} for {model}-{desc} (seed {seed})")
                continue

        if not predictions:
            ax.set_title(f"{desc} (No data)")
            continue

        x_labels = list(predictions.keys())
        x_pos = np.arange(len(x_labels))

        for i, assay in enumerate(x_labels):
            preds = predictions[assay]
            jitter = np.random.normal(loc=0, scale=0.05, size=len(preds))
            ax.scatter(np.full(len(preds), x_pos[i]) + jitter, preds,
                       alpha=0.6, s=10)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_title(f"{desc}")
        ax.set_xlabel("Assays")
        if j == 0:
            ax.set_ylabel("")

    fig.suptitle(f"Distribution for {model}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the combined figure
    outname = f"distribution_{model}_rf.png"
    plt.savefig(os.path.join(FIGUREPATH, outname), dpi=300)
    plt.close()