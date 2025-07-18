import os
import csv
import lazyqsar as lq
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)

# data = ADME(
#     name='bioavailability_ma',
# )
# split = data.get_split()
# smiles_train = list(split["train"]["Drug"])
# y_train = list(split["train"]["Y"])
# smiles_valid = list(split["valid"]["Drug"])
# y_valid = list(split["valid"]["Y"])

# NOTE: This version of tunetables supports arbitrary dataset size. This requires training.
# NOTE: If the dataset is small better to train them 1-4 epochs.
# NOTE: For big (>15k) one 10 epoch works fine. It has default early stopping of 4 epoch,
# NOTE: I used internally in tunetables, a PCA method to reduce the dimensions of fetures of mordred to 100


def load_datasets():
    with open("scalability_study/large.csv", "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        reader = list(reader)
    X = [row[1] for row in reader]
    y = [int(row[-1]) for row in reader]
    pos_count = len([pos for pos in y if pos == 1])
    neg_count = len([neg for neg in y if neg == 0])
    print(f"Pos count: {pos_count} | Neg count: {neg_count}")
    return train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)


# def fit():
#     model = lq.LazyBinaryQSAR(model_type="tunetables", descriptor_type="mordred")
#     print(f"Smiles: {smiles_train[2:3]}")
#     model.fit(smiles_train, y_train)
#     y_hat = model.predict_proba(smiles_valid)
#     fpr, tpr, _ = roc_curve(y_valid, y_hat)
#     print("AUROC", auc(fpr, tpr))

#     model.save_model("bioavailability_ma_mordered_one")


def fit_scale():
    X_train, X_valid, Y_train, Y_valid = load_datasets()
    Y_valid = [int(label) for label in Y_valid]
    print(f"Smiles sample: {X_train[1:2]} | lable sample: {Y_valid[0]}")
    model = lq.LazyBinaryQSAR(model_type="tunetables", descriptor_type="mordred")
    model.fit(X_train, Y_train)
    model.save_model("bioavailability_ma_mordered_10_ros_epoch_large")
    y_hat = model.predict_proba(X_valid)
    fpr, tpr, _ = roc_curve(Y_valid, y_hat)
    print("AUROC", auc(fpr, tpr))


def fit_scale_zsrf():
    X_train, X_valid, Y_train, Y_valid = load_datasets()
    Y_valid = [int(label) for label in Y_valid]
    print(f"Smiles sample: {X_train[1:2]} | lable sample: {Y_valid[0]}")
    model = lq.LazyBinaryQSAR(model_type="zsrandomforest", descriptor_type="morgan")
    model.fit(X_train, Y_train)
    model.save_model("bioavailability_ma_mordered_zsrf")
    y_hat = model.predict_proba(X_valid)
    fpr, tpr, _ = roc_curve(Y_valid, y_hat)
    print("AUROC", auc(fpr, tpr))


def fit_scale_xgboost():
    X_train, X_valid, Y_train, Y_valid = load_datasets()
    Y_valid = [int(label) for label in Y_valid]
    print(f"Smiles sample: {X_train[1:2]} | lable sample: {Y_valid[0]}")
    model = lq.LazyBinaryQSAR(model_type="xgboost", descriptor_type="morgan")
    model.fit(X_train, Y_train)
    model.save_model("bioavailability_ma_mordered_xgboost")
    y_hat = model.predict_proba(X_valid)
    fpr, tpr, _ = roc_curve(Y_valid, y_hat)
    print("AUROC", auc(fpr, tpr))


def predict():
    import time

    st = time.perf_counter()
    _, X_valid, _, Y_valid = load_datasets()
    Y_valid = [int(label) for label in Y_valid]
    # ds = list(zip(X_valid, Y_valid))
    # random.shuffle(ds)
    # X_valid, Y_valid = zip(*ds)
    # X_valid = list(X_valid)
    # Y_valid = list(Y_valid)
    print(len(X_valid), X_valid[0:2])
    model_dir = os.path.abspath("bioavailability_ma_mordered_10_ros_epoch_large")
    print(f"Absolute path of the model directory: {model_dir}")
    model = lq.LazyBinaryQSAR.load_model(model_dir)
    y_hat = model.predict_proba(X_valid)
    fpr, tpr, _ = roc_curve(Y_valid, y_hat)
    print("AUROC", auc(fpr, tpr))
    et = time.perf_counter()
    print(f"Inference done in {et - st:.3} seconds")


if __name__ == "__main__":
    # fit_scale()
    predict()
