
import os
from tdc.benchmark_group import admet_group
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FIGUREPATH =  "../figures"

clf_datasets = ["bioavailability_ma", "hia_hou", "pgp_broccatelli", "bbb_martins", "cyp2c9_veith","cyp2d6_veith",
                  "cyp3a4_veith", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
                  "cyp3a4_substrate_carbonmangels","herg","ames", "dili"]

reg_datasets = ['caco2_wang', 'lipophilicity_astrazeneca', 'solubility_aqsoldb', 'ppbr_az', 
                'vdss_lombardo',  'half_life_obach', 'clearance_microsome_az', 'clearance_hepatocyte_az', 'ld50_zhu']

group = admet_group(path = '../data/')

predictions_list = []
for seed in [1, 2, 3, 4, 5]:
    predictions = {}
    for a in clf_datasets:
        benchmark = group.get(a)
        name = benchmark['name']
        test = pd.read_csv(os.path.join("..", "data", "tdc_preds_morgan_100", "{}_test_{}.csv".format(a,seed)))
        #append predictions to predictions list
        predictions[name] = test["pred"]
        predictions_list.append(predictions)
results_morgan = group.evaluate_many(predictions_list)
print("CLF MORGAN")
print(results_morgan)

for seed in [1, 2, 3, 4, 5]:
    predictions = {}
    for a in clf_datasets:
        benchmark = group.get(a)
        name = benchmark['name']
        test = pd.read_csv(os.path.join("..", "data", "tdc_preds_eosce_100", "{}_test_{}.csv".format(a,seed)))
        #append predictions to predictions list
        predictions[name] = test["pred"]
        predictions_list.append(predictions)
results_eosce = group.evaluate_many(predictions_list)
print("CLF EOSCE")
print(results_eosce)

# Extract the keys and values from the dictionaries
assays = list(results_morgan.keys())
auroc_m = [x[0] for x in results_morgan.values()]
std_m = [x[1] for x in results_morgan.values()]
auroc_e = [x[0] for x in results_eosce.values()]
std_e = [x[1] for x in results_eosce.values()]

# Set the positions for the X-axis
x_pos = np.array(range(len(assays)))

# Set the colors for the plots
color1 = '#fad782'
color2 = '#faa08c'
offset = 0.05

# Create the plot
plt.figure(figsize=(10, 6))
plt.errorbar((x_pos-offset), auroc_m, yerr=std_m, fmt='o', color=color1, label='Morgan')
plt.errorbar((x_pos+offset), auroc_e, yerr=std_e, fmt='o', color=color2, label='EOSCE')
plt.xticks(x_pos-0.2, assays, rotation=45, ha="right")
plt.xlabel('Assay')
plt.ylabel('Model Performance')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGUREPATH, "clf_comparison_100.png"), dpi = 300)

predictions_list = []
for seed in [1, 2, 3, 4, 5]:
    predictions = {}
    for a in reg_datasets:
        benchmark = group.get(a)
        name = benchmark['name']
        test = pd.read_csv(os.path.join("..", "data", "tdc_preds_morgan_100", "{}_test_{}.csv".format(a,seed)))
        #append predictions to predictions list
        predictions[name] = test["pred"]
        predictions_list.append(predictions)
results_morgan = group.evaluate_many(predictions_list)
print("REG MORGAN")
print(results_morgan)

for seed in [1, 2, 3, 4, 5]:
    predictions = {}
    for a in reg_datasets:
        benchmark = group.get(a)
        name = benchmark['name']
        test = pd.read_csv(os.path.join("..", "data", "tdc_preds_eosce_100", "{}_test_{}.csv".format(a,seed)))
        #append predictions to predictions list
        predictions[name] = test["pred"]
        predictions_list.append(predictions)
results_eosce = group.evaluate_many(predictions_list)
print("REG EOSCE")
print(results_eosce)

# Extract the keys and values from the dictionaries
assays = list(results_morgan.keys())
auroc_m = [x[0] for x in results_morgan.values()]
std_m = [x[1] for x in results_morgan.values()]
auroc_e = [x[0] for x in results_eosce.values()]
std_e = [x[1] for x in results_eosce.values()]

# Set the positions for the X-axis
x_pos = np.array(range(len(assays)))

# Set the colors for the plots
color1 = '#fad782'
color2 = '#faa08c'
offset = 0.05

plt.figure(figsize=(10, 6))
plt.errorbar((x_pos-offset), auroc_m, yerr=std_m, fmt='o', color=color1, label='Morgan')
plt.errorbar((x_pos+offset), auroc_e, yerr=std_e, fmt='o', color=color2, label='EOSCE')
plt.xticks(x_pos-0.2, assays, rotation=45, ha="right")
plt.xlabel('Assay')
plt.ylabel('Model Performance')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGUREPATH, "reg_comparison_100.png"), dpi = 300)