import os
import lazyqsar as lq
from sklearn.metrics import roc_curve, auc
from tdc.benchmark_group import admet_group


DATAPATH = "../data"

clf_datasets = ["bioavailability_ma", "hia_hou", "pgp_broccatelli", "bbb_martins", "cyp2c9_veith","cyp2d6_veith",
                  "cyp3a4_veith", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
                  "cyp3a4_substrate_carbonmangels","herg","ames", "dili"]

# NOTE: This version of tunetables supports arbitrary dataset size. This requires training.
# NOTE: If the dataset is small better to train them 1-4 epochs. 
# NOTE: For big (>15k) one 10 epoch works fine. It has default early stopping of 4 epoch,
# NOTE: I used internally in tunetables, a PCA method to reduce the dimensions of fetures of mordred to 100


def get_data():
    group = admet_group(path = '../data/')
    names = group.dataset_names
    return names

def fit_clf(X, y, dataset, seed):
    model = lq.LazyBinaryQSAR(model_type="tunetables", descriptor_type="mordred") 
    model.fit(X, y)
    model.save_model(f"checkpoints_tunetables/{dataset}_{seed}")
    return model


if __name__ == '__main__':

    group = admet_group(path = '../data/')
    
    for seed in [1, 2, 3, 4, 5]:
        for a in clf_datasets:
            print(seed, a)
            benchmark = group.get(a) 
            name = benchmark['name']
            train_val, test = benchmark['train_val'], benchmark['test']
            model = fit_clf(train_val["Drug"], train_val["Y"], a, seed)
            model_dir = os.path.abspath(f"checkpoints_tunetables/{a}_{seed}")
            model = lq.LazyBinaryQSAR.load_model(model_dir=model_dir) 
            y_pred_test = model.predict_proba(test["Drug"])
            test["pred"] = y_pred_test[:,1]
            test["bin_pred"] = [0 if x < 0.5 else 1 for x in y_pred_test[:,1]]
            fpr, tpr, _ = roc_curve(test["Y"], y_pred_test[:,1])
            print("AUROC", auc(fpr, tpr))
            csv_path = os.path.join(DATAPATH, "tdc_preds_tunetable_mordred_100_tunetables")
            if not os.path.exists(csv_path):
                os.makedirs(csv_path, exist_ok=True)
            test.to_csv(os.path.join(csv_path, "{}_test_{}.csv".format(a,seed)), index=False)      
    