import pandas as pd
import os

clf_datasets = ["bioavailability_ma", "hia_hou", "pgp_broccatelli", "bbb_martins", "cyp2c9_veith","cyp2d6_veith",
                  "cyp3a4_veith", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
                  "cyp3a4_substrate_carbonmangels","herg","ames", "dili"]

for c in clf_datasets:
    df = pd.read_csv(os.path.join("admet_group", c, "train_val.csv"))
    pos = len(df[df["Y"]==1])
    prop = pos/len(df)
    print(c, prop)