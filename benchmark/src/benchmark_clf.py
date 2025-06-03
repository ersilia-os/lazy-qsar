import os
import sys
from tdc import utils
from tdc.benchmark_group import admet_group
import lazyqsar as lq

model_type = sys.argv[1]
desc = sys.argv[2]

DATAPATH = "../data"

clf_datasets = ["bioavailability_ma", "hia_hou", "pgp_broccatelli", "bbb_martins", "cyp2c9_veith","cyp2d6_veith",
                  "cyp3a4_veith", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
                  "cyp3a4_substrate_carbonmangels","herg","ames", "dili"]
clf_datasets = ["herg"]

def get_data():
    group = admet_group(path = '../data/')
    names = group.dataset_names
    return names

if __name__ == '__main__':

    group = admet_group(path = '../data/')
    for seed in [1,2,3,4,5]:
        for a in clf_datasets:
            print(seed, a)
            benchmark = group.get(a) 
            name = benchmark['name']
            train_val, test = benchmark['train_val'], benchmark['test']
            print(len(train_val), len(test))
            if model_type=="zeroshot" and len(train_val) > 1000:
                print("Skipping zeroshot for dataset with more than 1000 samples")
                continue
            model = lq.LazyBinaryQSAR(model_type=model_type, descriptor_type=desc)
            model.fit(train_val["Drug"], train_val["Y"])
            y_pred_test = model.predict_proba(test["Drug"])
            test["pred"] = y_pred_test
            save_path = os.path.join(DATAPATH, f"tdc_preds_{model_type}_{desc}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            test.to_csv(os.path.join(save_path, "{}_test_{}.csv".format(a,seed)), index=False)
    