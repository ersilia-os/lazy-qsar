import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
import sys

FIGUREPATH =  "../figures"
DATAPATH = "../data"

# Compare METHODS

descs = ["morgan", "mordred"]
models = ["xgboost", "xgboost_pca","randomforest","zsrandomforest"]


benchmark = {"bioavailability_ma":[0.748,0.033], "hia_hou":[0.989,0.001], "pgp_broccatelli":[0.938,0.002], 
             "bbb_martins":[0.916,0.001], "cyp2c9_veith":[0.859,0.001],"cyp2d6_veith":[0.790,0.001],
             "cyp3a4_veith":[0.916,0.000], "cyp2c9_substrate_carbonmangels":[0.441,0.033], 
             "cyp2d6_substrate_carbonmangels":[0.736,0.024],"cyp3a4_substrate_carbonmangels":[0.662,0.031],
             "herg":[0.880,0.002],"ames":[0.871,0.002], "dili":[0.925,0.005]}

colors = ["#50285a", "#fad782", "#faa08c", "#dca0dc", "#aa96fa"]


for desc in descs:
    combined_data = {}
    for model in models:
        file_path = os.path.join(DATAPATH, f"tdc_preds_{model}_{desc}", f"{model}_{desc}.json")
        with open(file_path, "r") as f:
            data = json.load(f)
            for assay, (mean, std) in data.items():
                if assay not in combined_data:
                    combined_data[assay] = {}
                combined_data[assay][f"{model}_mean"] = mean
                combined_data[assay][f"{model}_std"] = std

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(combined_data, orient="index")
    df = df.sort_index()
    assays = df.index.tolist()
    df["benchmark_mean"] = [benchmark[assay][0] for assay in assays]
    df["benchmark_std"] = [benchmark[assay][1] for assay in assays]
    x = np.arange(len(assays))  # X locations for each assay
    width = 0.1
    num_models = len(models) + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        mean = df[f"{model}_mean"].values
        std = df[f"{model}_std"].values
        offset_x = x + (i - num_models / 2) * width + width / 2
        ax.errorbar(
        offset_x, mean, yerr=std, fmt='o',
        label=model, color=colors[i],
        capsize=2, elinewidth=1, capthick=1
    )
    mean = df["benchmark_mean"].values
    std = df["benchmark_std"].values
    benchmark_idx = len(models)
    offset_x = x + (benchmark_idx - num_models / 2) * width + width / 2
    ax.errorbar(
        offset_x, mean, yerr=std, fmt='o',
        label="benchmark", color="#bee6b4",
        capsize=2, elinewidth=1, capthick=1
    )
    ax.set_ylabel("Performance metric (AUROC/AUPRC)")
    ax.set_title(f"Model Performance Across TDC Assays {desc}")
    ax.set_xticks(x+(width*(num_models/2)))
    ax.set_xticklabels(assays, rotation=45, ha="right")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGUREPATH, f"{desc}.png"), dpi = 300)

# Compare DESCRIPTORS
for model in models:
    combined_data = {}
    for desc in descs:
        file_path = os.path.join(DATAPATH, f"tdc_preds_{model}_{desc}", f"{model}_{desc}.json")
        with open(file_path, "r") as f:
            data = json.load(f)
            for assay, (mean, std) in data.items():
                if assay not in combined_data:
                    combined_data[assay] = {}
                combined_data[assay][f"{desc}_mean"] = mean
                combined_data[assay][f"{desc}_std"] = std

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(combined_data, orient="index")
    df = df.sort_index()
    assays = df.index.tolist()
    df["benchmark_mean"] = [benchmark[assay][0] for assay in assays]
    df["benchmark_std"] = [benchmark[assay][1] for assay in assays]

    x = np.arange(len(assays))
    width = 0.1
    fig, ax = plt.subplots(figsize=(10, 6))
    num_descs = len(descs) + 1

    for i, desc in enumerate(descs):
        mean = df[f"{desc}_mean"].values
        std = df[f"{desc}_std"].values
        offset_x = x + (i - num_descs / 2) * width + width / 2
        ax.errorbar(
            offset_x, mean, yerr=std, fmt='o',
            label=desc, color=colors[i],
            capsize=2, elinewidth=1, capthick=1
        )
    mean = df["benchmark_mean"].values
    std = df["benchmark_std"].values
    offset_x = x + (benchmark_idx - num_descs / 2) * width + width / 2
    ax.errorbar(
        offset_x, mean, yerr=std, fmt='o',
        label="benchmark", color="#bee6b4",
        capsize=2, elinewidth=1, capthick=1
    )

    ax.set_ylabel("Performance metric (AUROC/AUPRC)")
    ax.set_title("Model Performance Across TDC Assays")
    ax.set_xticks(x + width)
    ax.set_xticklabels(assays, rotation=45, ha="right")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGUREPATH, f"{model}.png"), dpi = 300)


# PCA vs BEST
"""
files = {
    "xgboost_morgan": os.path.join(DATAPATH, f"tdc_preds_xgboost_morgan", f"xgboost_morgan.json"),
    "xgboost_morgan_pca": os.path.join(DATAPATH, f"tdc_preds_xgboost_morgan_pca", f"xgboost_morgan.json"),
    "xgboost_mordred": os.path.join(DATAPATH, f"tdc_preds_xgboost_mordred", f"xgboost_mordred.json"),
    "xgboost_mordred_pca": os.path.join(DATAPATH, f"tdc_preds_xgboost_mordred_pca", f"xgboost_mordred.json")
}

combined_data = {}
for key, file_path in files.items():

    with open(file_path, "r") as f:
        data = json.load(f)
        for assay, (mean, std) in data.items():
            if assay not in combined_data:
                combined_data[assay] = {}
            combined_data[assay][f"{key}_mean"] = mean
            combined_data[assay][f"{key}_std"] = std

# Convert to DataFrame
df = pd.DataFrame.from_dict(combined_data, orient="index")
df = df.sort_index()
assays = df.index.tolist()
df["benchmark_mean"] = [benchmark[assay][0] for assay in assays]
df["benchmark_std"] = [benchmark[assay][1] for assay in assays]
x = np.arange(len(assays))  # X locations for each assay

width = 0.15
fig, ax = plt.subplots(figsize=(10, 6))

colors = ["#50285a", "#fad782", "#faa08c", "#dca0dc"]
for i, key in enumerate(files.keys()):
    mean = df[f"{key}_mean"].values
    std = df[f"{key}_std"].values
    ax.bar(x + i * width, mean, width, yerr=std, label=key, color= colors[i], capsize=4)
mean = df["benchmark_mean"].values
std = df["benchmark_std"].values
ax.bar(x + (i+1) * width, mean, width, yerr=std, label="benchmark", color= "#bee6b4", capsize=4)

ax.set_ylabel("Performance metric (AUROC/AUPRC)")
ax.set_title("Model Performance Across TDC Assays")
ax.set_xticks(x + width)
ax.set_xticklabels(assays, rotation=45, ha="right")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGUREPATH, "xgboost_pca.png"), dpi = 300)
"""