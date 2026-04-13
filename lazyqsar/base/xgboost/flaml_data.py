"""
FLAML XGBoost portfolio data.

Extracted verbatim from microsoft/FLAML (MIT license):
  flaml/default/xgboost/binary.json
  flaml/default/xgboost/regression.json

The lossguide ("xgboost") variant is used — it builds asymmetric trees via
grow_policy="lossguide" + max_leaves instead of max_depth.

1-NN algorithm (from FLAML's README):
  1. Compute query meta-features:
       [NumberOfInstances, NumberOfFeatures, NumberOfClasses,
        PercentageOfNumericFeatures]
  2. Normalize: q_norm = (q - center) / scale
  3. Find neighbor with smallest squared Euclidean distance to q_norm
  4. Use that neighbor's choice[0] as the portfolio index

The portfolio entry at that index contains the hyperparameters that FLAML
meta-learned to work best for datasets similar to the query.
"""

# ---------------------------------------------------------------------------
# Binary classification  (NumberOfClasses = 2)
# ---------------------------------------------------------------------------
BINARY = {
    "preprocessing": {
        "center": [18000.0, 28.0, 2.0, 0.7565217391304347],
        "scale":  [42124.0, 130.0, 1.0, 0.5714285714285715],
    },
    "portfolio": [
        {   # 0 – fabert
            "n_estimators": 319,
            "max_leaves": 1312,
            "min_child_weight": 0.001,
            "learning_rate": 0.01872379806270421,
            "subsample": 0.6890079660561895,
            "colsample_bylevel": 0.7551225121854014,
            "colsample_bytree": 0.7860755604500558,
            "reg_alpha": 0.17028752704343114,
            "reg_lambda": 1.4375743264564231,
        },
        {   # 1 – bng_lowbwt
            "n_estimators": 7902,
            "max_leaves": 49,
            "min_child_weight": 0.038063497848955595,
            "learning_rate": 0.0009765625,
            "subsample": 0.9357800695141445,
            "colsample_bylevel": 0.47031312177249246,
            "colsample_bytree": 0.9053386579586192,
            "reg_alpha": 1.5286102593845932,
            "reg_lambda": 18.96811296717419,
        },
        {   # 2 – pol
            "n_estimators": 13499,
            "max_leaves": 60,
            "min_child_weight": 0.008494221584011285,
            "learning_rate": 0.006955765856675575,
            "subsample": 0.5965241023754743,
            "colsample_bylevel": 0.590641168068946,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.2522240954379289,
            "reg_lambda": 5.351809144038808,
        },
        {   # 3 – Amazon_employee_access
            "n_estimators": 591,
            "max_leaves": 16651,
            "min_child_weight": 0.03356567864689129,
            "learning_rate": 0.002595066436678338,
            "subsample": 0.9114132805513452,
            "colsample_bylevel": 0.9503441844594458,
            "colsample_bytree": 0.5703338448066768,
            "reg_alpha": 0.010405212349127894,
            "reg_lambda": 0.05352660657433639,
        },
    ],
    "neighbors": [
        {"features": [1.196467571930491,   1.0923076923076922,  0.0, 0.4260869565217391],  "choice": [0, 3, 2, 1]},
        {"features": [11.096856898680088,  -0.16153846153846155, 0.0, -0.5739130434782609], "choice": [0, 2, 3, 1]},
        {"features": [8.658152122305575,   0.38461538461538464, 0.0, -0.7405797101449274], "choice": [2, 0, 1, 3]},
        {"features": [0.27281359794891274, -0.14615384615384616, 0.0, -1.3239130434782607], "choice": [3, 0, 2, 1]},
        {"features": [-0.4125676573924604, -0.1076923076923077,  0.0, -0.5739130434782609], "choice": [3, 1, 0, 2]},
        {"features": [0.6409647706770487,  1.5538461538461539,  0.0, 0.0],                 "choice": [1, 0, 2, 3]},
        {"features": [2.3515573069983855,  0.16923076923076924, 0.0, 0.4260869565217391],  "choice": [2, 0, 1, 3]},
        {"features": [0.6162045389801538,  -0.1076923076923077,  0.0, -0.5739130434782609], "choice": [1, 0, 2, 3]},
        {"features": [0.5386240622922799,  -0.09230769230769231, 0.0, -0.5582880434782608], "choice": [0, 1, 3, 2]},
        {"features": [-0.41133320672300827, -0.18461538461538463, 0.0, 0.4260869565217391], "choice": [2, 1, 0, 3]},
        {"features": [-0.31155635742094767, 12.36923076923077,  0.0, 0.3865087169129372],  "choice": [2, 1, 0, 3]},
        {"features": [-0.40594435476213087, -0.06153846153846154, 0.0, -0.7114130434782607], "choice": [0, 1, 2, 3]},
        {"features": [0.0,                  32.83076923076923,   0.0, 0.4260869565217391],  "choice": [0]},
        {"features": [1.6675766783781218,   0.0,                 0.0, 0.4260869565217391],  "choice": [2, 0, 1, 3]},
        {"features": [-0.36356946158959264, 0.8923076923076924,  0.0, -1.2266908212560386], "choice": [3, 1, 0, 2]},
        {"features": [-0.38225239768303104, -0.05384615384615385, 0.0, 0.4260869565217391], "choice": [3, 2, 0, 1]},
        {"features": [-0.3590352293229513,  0.06153846153846154, 0.0, -1.3239130434782607], "choice": [2, 0, 1, 3]},
        {"features": [0.3090399772101415,   0.6923076923076923,  0.0, -0.003997789240972687], "choice": [2, 0, 3, 1]},
        {"features": [-0.3118649700883107,  -0.17692307692307693, 0.0, 0.4260869565217391], "choice": [2, 0, 1, 3]},
        {"features": [0.0,                  32.83076923076923,   0.0, 0.4260869565217391],  "choice": [0, 3]},
        {"features": [-0.3178473079479632,  -0.06153846153846154, 0.0, 0.4260869565217391], "choice": [0, 3, 1, 2]},
    ],
}

# ---------------------------------------------------------------------------
# Regression  (NumberOfClasses = 0)
# ---------------------------------------------------------------------------
REGRESSION = {
    "preprocessing": {
        "center": [36691.0, 10.0, 0.0, 1.0],
        "scale":  [324551.25, 2.5, 1.0, 0.36111111111111116],
    },
    "portfolio": [
        {   # 0 – Albert
            "n_estimators": 6357,
            "max_leaves": 206,
            "min_child_weight": 1.9495322566288034,
            "learning_rate": 0.0068766724195393905,
            "subsample": 0.9451618245005704,
            "colsample_bylevel": 0.9030482524943064,
            "colsample_bytree": 0.9278972006416252,
            "reg_alpha": 0.01857648400903689,
            "reg_lambda": 6.021166480604588,
        },
        {   # 1 – mv
            "n_estimators": 23045,
            "max_leaves": 247,
            "min_child_weight": 0.004319397499079841,
            "learning_rate": 0.0032914413473281215,
            "subsample": 0.7334190564433234,
            "colsample_bylevel": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.03514226467919635,
            "reg_lambda": 1.2679661021665851,
        },
        {   # 2 – bng_echomonths
            "n_estimators": 1899,
            "max_leaves": 59,
            "min_child_weight": 0.013389019900720164,
            "learning_rate": 0.0028943401472847964,
            "subsample": 0.7808944208233943,
            "colsample_bylevel": 1.0,
            "colsample_bytree": 0.9999355357362375,
            "reg_alpha": 0.7905117773932884,
            "reg_lambda": 2.916897119216104,
        },
        {   # 3 – house_16H
            "n_estimators": 5611,
            "max_leaves": 61,
            "min_child_weight": 0.01070518287797225,
            "learning_rate": 0.005485127037677848,
            "subsample": 0.4713518256961299,
            "colsample_bylevel": 0.9777437906530106,
            "colsample_bytree": 0.9519335125615331,
            "reg_alpha": 0.03621564207188963,
            "reg_lambda": 1.8045765669466283,
        },
    ],
    "neighbors": [
        {"features": [0.0,                    0.0,   0.0, 0.0],                   "choice": [2, 3, 0, 1]},
        {"features": [-0.07492191140844474,   12.0,  0.0, 0.0],                   "choice": [0, 1, 3, 2]},
        {"features": [2.6600082421497375,     -0.4,  0.0, -0.923076923076923],    "choice": [3, 0, 2, 1]},
        {"features": [0.21039820367353385,    -0.4,  0.0, -2.4615384615384612],   "choice": [3, 2, 0, 1]},
        {"features": [-0.06453526215043079,   -0.4,  0.0, -0.923076923076923],    "choice": [2, 3, 0, 1]},
        {"features": [-0.026800081651203008,  -0.4,  0.0, -2.1538461538461537],   "choice": [2, 3, 0, 1]},
        {"features": [2.6600082421497375,      3.2,  0.0, -1.2307692307692306],   "choice": [1, 0, 3, 2]},
        {"features": [2.6600082421497375,      0.0,  0.0, -2.492307692307692],    "choice": [3, 0, 2, 1]},
        {"features": [0.3781868040871819,      0.0,  0.0, 0.0],                   "choice": [2, 3, 0, 1]},
        {"features": [0.0,                     0.0,  0.0, 0.0],                   "choice": [3, 0, 1, 2]},
        {"features": [-0.04987193856132121,    2.4,  0.0, 0.0],                   "choice": [3, 1, 0, 2]},
        {"features": [-0.04987193856132121,   -0.8,  0.0, 0.0],                   "choice": [2, 0, 1, 3]},
        {"features": [-0.0558155299047531,    -0.8,  0.0, 0.0],                   "choice": [0, 3, 1, 2]},
        {"features": [0.0,                     0.0,  0.0, -0.8307692307692308],   "choice": [1, 0, 3, 2]},
        {"features": [2.729362465866331,       0.0,  0.0, 0.0],                   "choice": [1, 0, 3, 2]},
        {"features": [-0.07145558675247746,   15.2,  0.0, 0.0],                   "choice": [0, 3, 1, 2]},
    ],
}
