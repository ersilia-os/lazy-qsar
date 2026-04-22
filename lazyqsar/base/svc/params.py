"""
Heuristic hyperparameter rules for SVC classification.

All rules are literature-backed for QSAR classification on molecular descriptors:

  Kernel selection
  ----------------
  Sparse integer-count fingerprints (Morgan / ECFP4, sparsity >= 0.85):
    kernel='linear'
    Tanimoto similarity is equivalent to the linear dot product on bit vectors;
    linear SVMs are state-of-the-art for fingerprint QSAR (Burbidge et al. 2001;
    Heikamp & Bajorath 2014; Sheridan 2016).

  Dense continuous descriptors (RDKit, CDDD, CheMeleon), n <= 5000:
    kernel='rbf'
    RBF captures non-linear structure in continuous feature spaces and
    typically outperforms the linear kernel when features are not dominated
    by bit counts (Burbidge 2001; Cherkassky & Ma 2004).

  Dense continuous + n > 5000 (fallback):
    kernel='linear'
    Kernel matrix is O(n^2) memory and cost; linear is tractable and still
    competitive at larger sample sizes.

  C (regularisation strength)
  ---------------------------
  Linear kernel:
    n < 500  → C = 0.1   (high regularisation; few samples, prevent overfit)
    500–2000 → C = 1.0   (standard; Heikamp & Bajorath 2014 optimal for ECFP4)
    >= 2000  → C = 10.0  (more data constrains the model; looser margin)

  RBF kernel — base value by n/p ratio (information density):
    n/p <  2  → C =   1.0  (underdetermined; strong regularisation)
    n/p 2–5   → C =  10.0
    n/p >= 5  → C = 100.0
    Scaled by min(1.0, sqrt(n / 1000)) to prevent C=100 + n=5000 overfitting.
    Capped at 100.0 regardless.

  gamma (RBF only)
  ----------------
  gamma='scale' = 1 / (n_features * Var(X)).
  Well-calibrated default for continuous descriptors (Caputo et al. 2002;
  matches Burbidge's 1/p rule when Var ≈ 1).

  class_weight
  ------------
  Always 'balanced' in QSAR due to common assay imbalance (Goh et al. 2017).

  max_iter / tol
  --------------
  max_iter = max(1000, min(10000, n * 5)): prevents infinite loops and scales
  with dataset size to allow sufficient convergence iterations.
  tol = 1e-3: sklearn default; fast convergence for portfolio evaluation.
"""

from __future__ import annotations

from .inspector import DatasetProfile


def get_params(profile: DatasetProfile) -> dict:
    """
    Return heuristic SVC hyperparameters for the given dataset profile.

    Parameters
    ----------
    profile : DatasetProfile
        Dataset statistics computed by inspector.inspect().

    Returns
    -------
    dict
        Complete parameter dict suitable for sklearn.svm.SVC or
        sklearn.svm.LinearSVC, plus the meta-key 'use_linear' (bool).
    """
    n = profile.n_samples
    p = profile.n_features
    n_p = profile.n_p_ratio

    # ------------------------------------------------------------------ kernel
    use_linear = (
        profile.is_sparse_counts
        or n > 5_000
    )

    # ------------------------------------------------------------------ C
    if use_linear:
        if n < 500:
            C = 0.1
        elif n < 2_000:
            C = 1.0
        else:
            C = 10.0
    else:  # RBF kernel
        if n_p < 2:
            C_base = 1.0
        elif n_p < 5:
            C_base = 10.0
        else:
            C_base = 100.0
        # Scale down for smaller n to prevent overfitting
        scale = min(1.0, (n / 1_000) ** 0.5)
        C = min(100.0, C_base * scale)

    # ------------------------------------------------- class weight / imbalance
    class_weight = "balanced"

    # ------------------------------------------------------ max_iter / tol
    max_iter = max(1_000, min(10_000, n * 5))
    tol = 1e-3

    params = {
        "C": round(C, 6),
        "class_weight": class_weight,
        "max_iter": max_iter,
        "tol": tol,
        "random_state": 42,
        "use_linear": use_linear,
    }

    if not use_linear:
        params["kernel"] = "rbf"
        params["gamma"] = "scale"
    # LinearSVC has no kernel/gamma parameters

    return params
