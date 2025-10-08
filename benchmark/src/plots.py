import pandas as pd
import os
import sys
import json
import numpy as np
from sklearn.metrics import roc_curve, auc

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

FIGUREPATH = os.path.join(root, "..", "figures")
PREDSPATH = os.path.join(root, "..", "preds")

from defaults import ADMET_CLF_TASKS, benchmark

c = {
    "chemeleon": "#50285a",
    "morgan": "#aa96fa",
}

def plot_roc_curve(ax, task, desc):
    test_results = pd.read_csv(os.path.join(
            PREDSPATH, f"tdc_preds_{desc}", f"{task}_test_pred.csv"
        ))
    fpr, tpr, _ = roc_curve(test_results["Y"], test_results["pred"])
    auc_score = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{desc} (AUC={auc_score:.2f})", color=c[desc])
    ax.legend()
    ax.set_title(task)
    ax.set_xlabel("")
    ax.set_ylabel("")

def plot_distribution(ax, task, desc):
    test_results = pd.read_csv(os.path.join(
                PREDSPATH, f"tdc_preds_{desc}", f"{task}_test_pred.csv"
            ))
    preds = test_results["pred"].tolist()
    jitter = np.random.normal(loc=0, scale=0.02, size=len(preds))
    ax.set_ylim(-0.05, 1.05)
    ax.scatter(jitter, preds, alpha=0.3, s=10, color= c[desc])
    ax.set_title(f"{task[:6] if len(task) > 6 else task}")
    ax.set_xticks([])
    ax.set_xticklabels("")
    ax.set_ylabel("")
    ax.set_xlabel(f"")
    