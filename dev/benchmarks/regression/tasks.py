"""Benchmark tasks: what each source contributes, in which units, and how it is split.

Every task ends up as ``(smiles, y)`` with ``y`` in log units, so that RMSE means the
same thing across tasks and a few extreme values cannot dominate it:

- TDC ADMET regression. Caco2, lipophilicity, solubility and LD50 are published in log
  units already. VDss, half-life and both clearances are raw positive quantities and are
  log10-transformed here (the convention of Polaris and of most ADMET papers). PPBR keeps
  human measurements only (the species TDC's ADMET group uses), in % bound.
- MoleculeACE: ``y [pEC50/pKi]`` with the dataset's own train/test split, which is
  stratified on activity cliffs.
- Biogen ADME: six endpoints, each already in log units; each is a task over the
  compounds that have a value for it.

Splits: ``random-<seed>`` (80/20) and ``scaffold`` (Bemis-Murcko groups, largest first
into train until 80%, so test scaffolds are unseen). MoleculeACE tasks use their
``provided`` split in place of the scaffold split.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, ".cache", "raw")
FEATURES = os.path.join(HERE, ".cache", "features")

_TDC = {
    # task: (file stem, smiles column, log10-transform?)
    "tdc_caco2": ("tdc_caco2_wang", "Drug", False),
    "tdc_lipophilicity": ("tdc_lipophilicity_astrazeneca", "Drug", False),
    "tdc_solubility": ("tdc_solubility_aqsoldb", "Drug", False),
    "tdc_ppbr": ("tdc_ppbr_az", "Drug", False),
    "tdc_ld50": ("tdc_ld50_zhu", "X", False),
    "tdc_vdss": ("tdc_vdss_lombardo", "X", True),
    "tdc_half_life": ("tdc_half_life_obach", "X", True),
    "tdc_clearance_hepatocyte": ("tdc_clearance_hepatocyte_az", "X", True),
    "tdc_clearance_microsome": ("tdc_clearance_microsome_az", "X", True),
}
_BIOGEN = {
    "biogen_hlm_clint": "LOG HLM_CLint (mL/min/kg)",
    "biogen_rlm_clint": "LOG RLM_CLint (mL/min/kg)",
    "biogen_mdr1_er": "LOG MDR1-MDCK ER (B-A/A-B)",
    "biogen_solubility": "LOG SOLUBILITY PH 6.8 (ug/mL)",
    "biogen_ppb_human": "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)",
    "biogen_ppb_rat": "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)",
}


def task_names() -> list[str]:
    """Every task, TDC first, then Biogen, then MoleculeACE."""
    ace = sorted(
        f[len("moleculeace_") : -len(".csv")]
        for f in os.listdir(RAW)
        if f.startswith("moleculeace_") and f.endswith(".csv")
    )
    return list(_TDC) + list(_BIOGEN) + [f"ace_{a}" for a in ace]


def _read_tdc(stem, smiles_col, log10):
    path = os.path.join(RAW, f"{stem}.tab")
    smiles, y = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if "Species" in row and row["Species"] != "Homo sapiens":
                continue
            value = float(row["Y"])
            if log10:
                if value <= 0:
                    continue
                value = np.log10(value)
            smiles.append(row[smiles_col])
            y.append(value)
    return smiles, np.asarray(y), None


def _read_biogen(column):
    smiles, y = [], []
    with open(os.path.join(RAW, "biogen_adme.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row[column] in ("", "nan", "NaN"):
                continue
            smiles.append(row["SMILES"])
            y.append(float(row[column]))
    return smiles, np.asarray(y), None


def _read_ace(name):
    smiles, y, split = [], [], []
    with open(os.path.join(RAW, f"moleculeace_{name}.csv"), newline="") as f:
        for row in csv.DictReader(f):
            smiles.append(row["smiles"])
            y.append(float(row["y [pEC50/pKi]"]))
            split.append(row["split"])
    return smiles, np.asarray(y), np.asarray(split)


def load_raw(task: str):
    """``(smiles, y, provided_split_or_None)`` straight from the source file."""
    if task in _TDC:
        return _read_tdc(*_TDC[task])
    if task in _BIOGEN:
        return _read_biogen(_BIOGEN[task])
    if task.startswith("ace_"):
        return _read_ace(task[len("ace_") :])
    raise KeyError(task)


def load(task: str, descriptor: str = "morgan"):
    """``(smiles, y, X, provided_split)`` with features cached on disk.

    Molecules the descriptor cannot featurize (an all-NaN row) are dropped, so every
    configuration is compared on exactly the same rows.
    """
    os.makedirs(FEATURES, exist_ok=True)
    cache = os.path.join(FEATURES, f"{task}__{descriptor}.npz")
    if os.path.isfile(cache):
        d = np.load(cache, allow_pickle=False)
        split = d["split"] if d["split"].size else None
        return list(d["smiles"]), d["y"], d["X"], split
    from lazyqsar.registry import get_descriptor_type

    smiles, y, split = load_raw(task)
    X = np.asarray(
        get_descriptor_type(descriptor)().transform(smiles), dtype=np.float32
    )
    keep = ~np.isnan(X).all(axis=1)
    smiles = [s for s, k in zip(smiles, keep) if k]
    y, X = y[keep], X[keep]
    split = split[keep] if split is not None else None
    np.savez_compressed(
        cache,
        smiles=np.asarray(smiles),
        y=y,
        X=X,
        split=split if split is not None else np.asarray([], dtype=str),
    )
    return smiles, y, X, split


def _scaffold_split(smiles, frac_train=0.8):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else s
        groups[key].append(i)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    n_train = int(round(frac_train * len(smiles)))
    train = []
    for g in ordered:
        if len(train) + len(g) <= n_train or not train:
            train.extend(g)
    train = np.asarray(sorted(train))
    test = np.setdiff1d(np.arange(len(smiles)), train)
    return train, test


def splits(task, smiles, provided, n_random=5):
    """``[(name, train_idx, test_idx)]``: random repeats plus one structural split."""
    n = len(smiles)
    out = []
    for seed in range(n_random):
        perm = np.random.default_rng(seed).permutation(n)
        cut = int(round(0.8 * n))
        out.append((f"random-{seed}", np.sort(perm[:cut]), np.sort(perm[cut:])))
    if provided is not None:
        out.append(
            (
                "provided",
                np.flatnonzero(provided == "train"),
                np.flatnonzero(provided == "test"),
            )
        )
    else:
        tr, te = _scaffold_split(smiles)
        out.append(("scaffold", tr, te))
    return out
