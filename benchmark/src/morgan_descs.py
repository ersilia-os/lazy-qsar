import os
from tdc.benchmark_group import admet_group
import lazyqsar as lq


DATAPATH = "../data"

clf_datasets = ["bioavailability_ma", "hia_hou", "pgp_broccatelli", "bbb_martins", "cyp2c9_veith","cyp2d6_veith",
                  "cyp3a4_veith", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
                  "cyp3a4_substrate_carbonmangels","herg","ames", "dili"]

reg_datasets = ['caco2_wang', 'lipophilicity_astrazeneca', 'solubility_aqsoldb', 'ppbr_az', 
                'vdss_lombardo',  'half_life_obach', 'clearance_microsome_az', 'clearance_hepatocyte_az', 'ld50_zhu']


def get_data():
    group = admet_group(path = '../data/')
    names = group.dataset_names
    return names

def fit_clf(X, y):
    model = lq.MorganBinaryClassifier(time_budget_sec = 600, estimator_list = ["rf"], reduced=True) 
    model.fit(X, y)
    return model

def fit_reg(X,y):
    model = lq.MorganRegressor(time_budget_sec = 600, estimator_list = ["rf"], reduced=True) 
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
            test.to_csv(os.path.join(DATAPATH, "tdc_preds_morgan_100", "{}_test_{}.csv".format(a,seed)), index=False)
    
    for seed in [1, 2, 3, 4, 5]:
        for a in reg_datasets:
            print(seed, a)
            benchmark = group.get(a) 
            name = benchmark['name']
            train_val, test = benchmark['train_val'], benchmark['test']
            model = fit_reg(train_val["Drug"], train_val["Y"])
            y_pred_test = model.predict(test["Drug"])
            test["pred"] = y_pred_test
            test.to_csv(os.path.join(DATAPATH, "tdc_preds_morgan_100", "{}_test_{}.csv".format(a,seed)), index=False)