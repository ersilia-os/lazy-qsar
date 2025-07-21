ADMET_CLF_TASKS = {
    "bioavailability_ma": {
        "name": "Bioavailability_Ma",
        "metric": "roc-auc"
    },
    "hia_hou": {
        "name": "HIA_Hou",
        "metric": "roc-auc"
    },
    "pgp_broccatelli": {
        "name": "Pgp_Broccatelli",
        "metric": "roc-auc"
    },
    "bbb_martins": {
        "name": "BBB_Martins",
        "metric": "roc-auc"
    },
    "cyp2c9_veith": {
        "name": "CYP2C9_Veith",
        "metric": "pr-auc"
    },
    "cyp2d6_veith": {
        "name": "CYP2D6_Veith",
        "metric": "pr-auc"
    },
    "cyp3a4_veith": {
        "name": "CYP3A4_Veith",
        "metric": "pr-auc"
    },
    "cyp2c9_substrate_carbonmangels": {
        "name": "CYP2C9_Substrate_CarbonMangels",
        "metric": "pr-auc"
    },
    "cyp2d6_substrate_carbonmangels": {
        "name": "CYP2D6_Substrate_CarbonMangels",
        "metric": "pr-auc"
    },
    "cyp3a4_substrate_carbonmangels": {
        "name": "CYP3A4_Substrate_CarbonMangels",
        "metric": "roc-auc"
    },
    "herg": {
        "name": "hERG",
        "metric": "roc-auc"
    },
    "ames": {
        "name": "AMES",
        "metric": "roc-auc"
    },
    "dili": {
        "name": "DILI",
        "metric": "roc-auc"
    }
}

clf_datasets = [
    "bioavailability_ma",
    "hia_hou",
    "pgp_broccatelli",
    "bbb_martins",
    "cyp2c9_veith",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "herg",
    "ames",
    "dili",
]

benchmark = {
    "bioavailability_ma": [0.748, 0.033],
    "hia_hou": [0.989, 0.001],
    "pgp_broccatelli": [0.938, 0.002],
    "bbb_martins": [0.916, 0.001],
    "cyp2c9_veith": [0.859, 0.001],
    "cyp2d6_veith": [0.790, 0.001],
    "cyp3a4_veith": [0.916, 0.000],
    "cyp2c9_substrate_carbonmangels": [0.441, 0.033],
    "cyp2d6_substrate_carbonmangels": [0.736, 0.024],
    "cyp3a4_substrate_carbonmangels": [0.662, 0.031],
    "herg": [0.880, 0.002],
    "ames": [0.871, 0.002],
    "dili": [0.925, 0.005],
}