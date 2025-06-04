import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
import sys

FIGUREPATH =  "../figures"
DATAPATH = "../data"

desc = sys.argv[1]
models = ["xgboost", "tunetables", "zeroshot"]
combined_data = {}

benchmark = {"bioavailability_ma":[0.748,0.033], "hia_hou":[0.989,0.001], "pgp_broccatelli":[0.938,0.002], 
             "bbb_martins":[0.916,0.001], "cyp2c9_veith":[0.859,0.001],"cyp2d6_veith":[0.790,0.001],
             "cyp3a4_veith":[0.916,0.000], "cyp2c9_substrate_carbonmangels":[0.441,0.033], 
             "cyp2d6_substrate_carbonmangels":[0.736,0.024],"cyp3a4_substrate_carbonmangels":[0.662,0.031],
             "herg":[0.880,0.002],"ames":[0.871,0.002], "dili":[0.925,0.005]}

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
print(df.head(10))
x = np.arange(len(assays))  # X locations for each assay

width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

colors = ["#50285a", "#fad782", "#faa08c"]
for i, model in enumerate(models):
    mean = df[f"{model}_mean"].values
    std = df[f"{model}_std"].values
    ax.bar(x + i * width, mean, width, yerr=std, label=model, color= colors[i], capsize=4)
mean = df["benchmark_mean"].values
std = df["benchmark_std"].values
ax.bar(x + (i+1) * width, mean, width, yerr=std, label="benchmark", color= "#bee6b4", capsize=4)

ax.set_ylabel("Performance metric (AUROC/AUPRC)")
ax.set_title("Model Performance Across TDC Assays")
ax.set_xticks(x + width)
ax.set_xticklabels(assays, rotation=45, ha="right")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGUREPATH, f"{desc}.png"), dpi = 300)