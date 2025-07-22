import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import json

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

from defaults import benchmark

FIGUREPATH = "../figures"
DATAPATH = "../data"
PREDSPATH = "../predictions"

descs = ["chemeleon", "morgan"]
models = ["logistic_regression", "random_forest", "tune_tables"]

colors = ["#50285a", "#fad782", "#aa96fa"]

for desc in descs:
    combined_data = {}
    available_models = []
    for model in models:
        try:
            file_path = os.path.join(
                PREDSPATH, f"tdc_preds_{model}_{desc}", f"{model}_{desc}.json"
            )
            with open(file_path, "r") as f:
                data = json.load(f)
                for k, v in data.items():
                    if k not in combined_data:
                        combined_data[k] = {}
                    combined_data[k][f"{model}_performance"] = v
            available_models += [model]
        except:
            continue
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
    for i, model in enumerate(available_models):
        perf = df[f"{model}_performance"].values
        offset_x = x + (i - num_models / 2) * width + width / 2
        ax.errorbar(
            offset_x,
            perf,
            fmt="o",
            label=model,
            color=colors[i],
            capsize=2,
            elinewidth=1,
            capthick=1,
        )
    mean = df["benchmark_mean"].values
    std = df["benchmark_std"].values
    benchmark_idx = len(models)
    offset_x = x + (benchmark_idx - num_models / 2) * width + width / 2
    ax.errorbar(
        offset_x,
        mean,
        yerr=std,
        fmt="o",
        label="benchmark",
        color="#bee6b4",
        capsize=2,
        elinewidth=1,
        capthick=1,
    )
    ax.set_ylabel("Performance metric (AUROC/AUPRC)")
    ax.set_title(f"Model Performance Across TDC Assays {desc}")
    ax.set_xticks(x + (width * (num_models / 2)))
    ax.set_xticklabels(assays, rotation=45, ha="right")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGUREPATH, f"{desc}.png"), dpi=300)

# Compare DESCRIPTORS
for model in models:
    combined_data = {}
    available_descs = []
    for desc in descs:
        try:
            file_path = os.path.join(
                PREDSPATH, f"tdc_preds_{model}_{desc}", f"{model}_{desc}.json"
            )
            with open(file_path, "r") as f:
                data = json.load(f)
                for k, v in data.items():
                    if k not in combined_data:
                        combined_data[k] = {}
                    combined_data[k][f"{desc}_performance"] = v
            available_descs += [desc]
        except:
            continue
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

    for i, desc in enumerate(available_descs):
        perf = df[f"{desc}_performance"].values
        offset_x = x + (i - num_descs / 2) * width + width / 2
        ax.errorbar(
            offset_x,
            perf,
            fmt="o",
            label=desc,
            color=colors[i],
            capsize=2,
            elinewidth=1,
            capthick=1,
        )
    mean = df["benchmark_mean"].values
    std = df["benchmark_std"].values
    offset_x = x + (benchmark_idx - num_descs / 2) * width + width / 2
    ax.errorbar(
        offset_x,
        mean,
        yerr=std,
        fmt="o",
        label="benchmark",
        color="#bee6b4",
        capsize=2,
        elinewidth=1,
        capthick=1,
    )

    ax.set_ylabel("Performance metric (AUROC/AUPRC)")
    ax.set_title("Model Performance Across TDC Assays")
    ax.set_xticks(x + width)
    ax.set_xticklabels(assays, rotation=45, ha="right")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGUREPATH, f"{model}.png"), dpi=300)
