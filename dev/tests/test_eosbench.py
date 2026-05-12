# LazyQSAR benchmark tests on external datasets.
#
# Prerequisites — before running this script:
#
# 1. Clone and install eosbench:
#    git clone https://github.com/ersilia-os/eosbench
#    cd eosbench && pip install -e .
#
# 2. Create a datasets directory and cd into it:
#    mkdir datasets && cd datasets
#
# 3. Fetch all benchmark datasets:
#    eosbench fetch --source chembl --dataset chembl4649948 --featurization none
#    eosbench fetch --source chembl --dataset chembl4659961 --featurization none
#    eosbench fetch --source tdc --dataset ames --featurization none
#    eosbench fetch --source tdc --dataset bbb_martins --featurization none
#    eosbench fetch --source tdc --dataset bioavailability_ma --featurization none
#    eosbench fetch --source tdc --dataset carcinogens_lagunin --featurization none
#    eosbench fetch --source tdc --dataset clintox --featurization none
#    eosbench fetch --source tdc --dataset cyp1a2_veith --featurization none
#    eosbench fetch --source tdc --dataset cyp2c19_veith --featurization none
#    eosbench fetch --source tdc --dataset cyp2c9_substrate_carbonmangels --featurization none
#    eosbench fetch --source tdc --dataset cyp2c9_veith --featurization none
#    eosbench fetch --source tdc --dataset cyp2d6_substrate_carbonmangels --featurization none
#    eosbench fetch --source tdc --dataset cyp2d6_veith --featurization none
#    eosbench fetch --source tdc --dataset cyp3a4_substrate_carbonmangels --featurization none
#    eosbench fetch --source tdc --dataset cyp3a4_veith --featurization none
#    eosbench fetch --source tdc --dataset dili --featurization none
#    eosbench fetch --source tdc --dataset herg --featurization none
#    eosbench fetch --source tdc --dataset hia_hou --featurization none
#    eosbench fetch --source tdc --dataset pgp_broccatelli --featurization none
#    eosbench fetch --source tdc --dataset skin_reaction --featurization none

import argparse
import json
import os

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import lazyqsar
from lazyqsar.qsar import LazyClassifierQSAR

lazyqsar.set_verbosity(True)


def load_dataset(dataset_dir, source=None, dataset=None):
    data = pd.read_csv(os.path.join(dataset_dir, "data.csv"))
    folds = pd.read_csv(os.path.join(dataset_dir, "folds.csv"))["fold"].values

    smiles = data["smiles"].tolist()
    # ChEMBL uses "value"; TDC uses "activity"
    activity_col = "activity" if "activity" in data.columns else "value"
    y = data[activity_col].values

    metadata_path = os.path.join(dataset_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {
            "source": source,
            "dataset": dataset,
            "featurizer": None,
            "n_samples": int(len(y)),
            "n_positives": int((y == 1).sum()),
            "n_negatives": int((y == 0).sum()),
            "auroc_mean": 0,
            "auroc_std": 0,
            "aupr_mean": 0,
            "aupr_std": 0,
        }

    return smiles, y, folds, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets_dir",
        required=True,
        help=(
            "Root directory of eosbench datasets. "
            "Expected layout: <datasets_dir>/<source>/<task_type>/<dataset>/ "
            "where each leaf contains data.csv (columns: smiles, activity|value), "
            "folds.csv (column: fold), and optionally metadata.json."
        ),
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory where output CSVs are written."
    )
    parser.add_argument(
        "--source", required=True, help="Dataset source (e.g. tdc, chembl)."
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name (e.g. ames, herg)."
    )
    parser.add_argument(
        "--fold", required=True, type=int, help="Fold index to evaluate."
    )
    parser.add_argument(
        "--task_type",
        default="classification",
        help="Task type (default: classification).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(
        args.output_dir, f"{args.source}_{args.dataset}_fold{args.fold}.csv"
    )
    if os.path.exists(out_path) and not args.force:
        print(f"Output already exists, skipping: {out_path}")
        exit(0)

    dataset_dir = os.path.join(
        args.datasets_dir, args.source, args.task_type, args.dataset
    )
    smiles, y, folds, metadata = load_dataset(
        dataset_dir, source=args.source, dataset=args.dataset
    )
    n_folds = folds.max() + 1

    print(f"{args.source}/{args.dataset}  fold={args.fold}/{n_folds}  n={len(smiles)}")

    is_test = folds == args.fold
    is_train = ~is_test

    smiles_train = [s for s, flag in zip(smiles, is_train) if flag]
    y_train = y[is_train]
    smiles_test = [s for s, flag in zip(smiles, is_test) if flag]
    y_test = y[is_test]

    pos_rate_test = float(y_test.mean())

    print(
        f"  train={len(smiles_train)} ({int(y_train.sum())} pos)"
        f"  test={len(smiles_test)} ({int(y_test.sum())} pos)"
    )

    model = LazyClassifierQSAR(mode="slow")
    model.fit(smiles_train, y_train)

    proba = model.predict_proba(smiles_test)[:, 1]
    auroc = roc_auc_score(y_test, proba)
    aupr = average_precision_score(y_test, proba)
    aupr_baseline = pos_rate_test
    aupr_ratio = aupr / aupr_baseline if aupr_baseline > 0 else float("nan")

    print(
        f"  AUROC={auroc:.4f}  AUPR={aupr:.4f}  (baseline={aupr_baseline:.4f}  ratio={aupr_ratio:.2f}x)"
    )

    row = {
        "source": args.source,
        "dataset": args.dataset,
        "fold": args.fold,
        "n_folds": n_folds,
        "n_train": len(smiles_train),
        "n_test": len(smiles_test),
        "n_pos_train": int(y_train.sum()),
        "n_neg_train": int((y_train == 0).sum()),
        "n_pos_test": int(y_test.sum()),
        "n_neg_test": int((y_test == 0).sum()),
        "auroc": round(auroc, 4),
        "aupr": round(aupr, 4),
        "aupr_baseline": round(aupr_baseline, 4),
        "aupr_ratio": round(aupr_ratio, 4),
        "auroc_ref_mean": metadata.get("auroc_mean"),
        "auroc_ref_std": metadata.get("auroc_std"),
        "aupr_ref_mean": metadata.get("aupr_mean"),
        "aupr_ref_std": metadata.get("aupr_std"),
    }

    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"  → saved to {out_path}")
