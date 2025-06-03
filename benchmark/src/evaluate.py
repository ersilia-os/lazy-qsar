
import os
import sys
import json
from tdc.benchmark_group import admet_group
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FIGUREPATH =  "../figures"
DATAPATH = "../data"

model_type = sys.argv[1]
desc = sys.argv[2]

clf_datasets = ["bioavailability_ma", "hia_hou", "pgp_broccatelli", "bbb_martins", "cyp2c9_veith","cyp2d6_veith",
                  "cyp3a4_veith", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
                  "cyp3a4_substrate_carbonmangels","herg","ames", "dili"]

group = admet_group(path = '../data/')

predictions_list = []
for seed in [1, 2, 3, 4, 5]:
    predictions = {}
    for a in clf_datasets:
        benchmark = group.get(a)
        name = benchmark['name']
        try:
            test = pd.read_csv(os.path.join(DATAPATH, f"tdc_preds_{model_type}_{desc}", "{}_test_{}.csv".format(a,seed)))
        except:
            print(f"Skipping {a} for seed {seed} as the file does not exist.")
            continue
        predictions[name] = test["pred"]
        predictions_list.append(predictions)
results = group.evaluate_many(predictions_list)
print(results)

results_file = f"{model_type}_{desc}.json"

with open(os.path.join(DATAPATH,f"tdc_preds_{model_type}_{desc}", results_file), "w") as f:
    json.dump(results, f, indent=2)