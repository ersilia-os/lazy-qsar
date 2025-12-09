from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def load_dataset(dataset_name):
    with open("data/{0}_train.csv".format(dataset_name), "r") as f:
        reader = csv.reader(f)
        smiles_train = []
        y_train = []
        next(reader)
        for row in reader:
            smiles_train += [row[0]]
            y_train += [int(row[1])]
    print("Number of training samples: {0}".format(len(y_train)))

    with open("data/{0}_test.csv".format(dataset_name), "r") as f:
        reader = csv.reader(f)
        smiles_test = []
        y_test = []
        next(reader)
        for row in reader:
            smiles_test += [row[0]]
            y_test += [int(row[1])]
    print("Number of testing samples: {0}".format(len(y_test)))
    return smiles_train, y_train, smiles_test, y_test


def train_and_eval_rf(
    smiles_train,
    y_train,
    smiles_test,
    y_test,
    radius=2,
    n_bits=2048,
    n_estimators=300,
    random_state=42,
):
    """Train RF on Morgan fingerprints and evaluate on the test set."""

    def featurize(smiles_list):
        X = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
        keep = []
        for i, s in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                keep.append(False)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X[i] = arr
            keep.append(True)
        keep = np.array(keep, dtype=bool)
        return X[keep], keep

    # Featurize
    X_train, keep_tr = featurize(smiles_train)
    y_train = np.asarray(y_train)[keep_tr]
    X_test, keep_te = featurize(smiles_test)
    y_test = np.asarray(y_test)[keep_te]

    # Train
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc = float("nan")

    print(f"Accuracy: {acc:.4f} | ROC AUC: {roc:.4f}")
    return clf


smiles_train, y_train, smiles_test, y_test = load_dataset("bioavailability_ma")
clf = train_and_eval_rf(smiles_train, y_train, smiles_test, y_test)
