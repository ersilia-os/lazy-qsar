import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
import sys

FIGUREPATH =  "../figures"
DATAPATH = "../data"

desc = sys.argv[1]
models = ["xgboost", "zeroshot"]
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
x = np.arange(len(assays))  # X locations for each assay

width = 0.25
fig, ax = plt.subplots(figsize=(10, 6))

colors = ["#50285a", "#fad782", "#faa08c"]
for i, model in enumerate(models):
    mean = df[f"{model}_mean"].values
    std = df[f"{model}_std"].values
    ax.bar(x + i * width, mean, width, yerr=std, label=model, color= colors[i], capsize=4)

ax.set_ylabel("Performance metric (AUROC/AUPRC)")
ax.set_title("Model Performance Across TDC Assays")
ax.set_xticks(x + width)
ax.set_xticklabels(assays, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGUREPATH, f"{desc}.png"), dpi = 300)