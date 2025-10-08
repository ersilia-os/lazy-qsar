import os
import sys
import json
import pandas as pd
import stylia as st
from stylia import TWO_COLUMNS_WIDTH, ONE_COLUMN_WIDTH
import matplotlib.pyplot as plt


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

from defaults import ADMET_CLF_TASKS
from fit_eval import train_eval
from plots import plot_roc_curve, plot_distribution

PREDSPATH = os.path.join(root, "..", "preds")
FIGUREPATH = os.path.join(root, "..", "figures")

if not os.path.exists(FIGUREPATH):
    os.mkdir(FIGUREPATH)

descs = ["morgan", "chemeleon"]

"""
for desc in descs: 
    predictions = {}
    for k,v in ADMET_CLF_TASKS.items():
        performance = train_eval(k, desc, v["metric"])
        predictions[k] = performance
    results_file = f"clf_benchmark_{desc}.json"
    with open(
        os.path.join(PREDSPATH, f"tdc_preds_{desc}", results_file), "w"
    ) as f:
        json.dump(predictions, f, indent=2)
"""
fig, axs = st.create_figure(7, 2, width=ONE_COLUMN_WIDTH, height=TWO_COLUMNS_WIDTH)
for i, task in enumerate(ADMET_CLF_TASKS.keys()):
    ax = axs.next()
    for desc in descs:
        plot_roc_curve(ax, task, desc)
plt.tight_layout()
plt.savefig(os.path.join(FIGUREPATH, "auroc.png"), dpi=300)

fig, axs = st.create_figure(1,13,width=TWO_COLUMNS_WIDTH, height=ONE_COLUMN_WIDTH)
for i, task in enumerate(ADMET_CLF_TASKS.keys()):
    ax = axs.next()
    for desc in descs:
        plot_distribution(ax, task, desc)
plt.tight_layout()
plt.savefig(os.path.join(FIGUREPATH, "distribution.png"), dpi=300)



