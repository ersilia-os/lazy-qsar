import os
import pandas as pd

from sklearn.metrics import roc_curve, auc
from tdc.benchmark_group import admet_group
import lazyqsar as lq


DATAPATH = "../data"

clf_datasets = ["bioavailability_ma", "hia_hou"] 
# NOTE: Zero-shot only supports dataset size less or equals to 1000
# NOTE: You can batch them and aggregate the result as well
# NOTE: I used internally in tunetables, a PCA method to reduce the dimensions of fetures of mordred to 100
# NOTE: Each prediction run results a slight different results in accuracy 
# [this is inrentional, a random permutation in the label class yileds different results and with ensemble config, the results are aggregated]

def get_data():
    group = admet_group(path = '../data/')
    names = group.dataset_names
    return names

def fit_clf(X, y):
    model = lq.LazyBinaryQSAR(model_type="zeroshot", descriptor_type="mordred") 
    model.fit(X, y)
    return model


if __name__ == '__main__':

    group = admet_group(path = '../data/')
    
    for seed in [1, 2, 3, 4, 5]:
        for a in clf_datasets:
            print(seed, a)
            benchmark = group.get(a) 
            name = benchmark['name']
            train_val, test = benchmark['train_val'], benchmark['test']
            model = fit_clf(train_val["Drug"], train_val["Y"])
            y_pred_test = model.predict_proba(test["Drug"])
            test["pred"] = y_pred_test[:,1]
            test["bin_pred"] = [0 if x < 0.5 else 1 for x in y_pred_test[:,1]]
            fpr, tpr, _ = roc_curve(test["Y"], y_pred_test[:,1])
            print("AUROC", auc(fpr, tpr))
            csv_path = os.path.join(DATAPATH, "tdc_preds_tunetable_mordred_100_zeroshot")
            if not os.path.exists(csv_path):
                os.makedirs(csv_path, exist_ok=True)
            test.to_csv(os.path.join(csv_path, "{}_test_{}.csv".format(a,seed)), index=False)    
    