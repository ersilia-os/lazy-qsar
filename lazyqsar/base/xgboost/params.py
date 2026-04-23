"""
Zero-shot XGBoost hyperparameter selection.

All rules are derived from:
  - Friedman (2001) "Greedy Function Approximation: A Gradient Boosting Machine"
  - Chen & Guestrin (2016) XGBoost paper (arXiv:1603.02754)
  - XGBoost official documentation and parameter tuning notes
  - Winkelmolen et al. (2020) "Practical and Sample Efficient Zero-Shot HPO"
    (arXiv:2007.13382)
  - Lima Marinho et al. (2024) "Optimization on selecting XGBoost hyperparameters
    using meta-learning" (Expert Systems)
  - Sommer et al. (2019) "Learning to Tune XGBoost with XGBoost"
    (arXiv:1909.07218)
  - Hutter et al. (2014) fANOVA hyperparameter importance analysis (ICML)
  - Grinsztajn et al. (2022) "Why do tree-based models still outperform deep
    learning on tabular data?" NeurIPS (arXiv:2207.08815) — search space priors
  - Probst et al. (2023) "Practical guidelines for the use of gradient boosting
    for molecular property prediction" J. Cheminformatics — QSAR-specific priors;
    ranked learning_rate, scale_pos_weight, gamma as the three most impactful
    parameters; recommends lr 0.01-0.05 and min_child_weight 5-20 for fingerprints
  - Sheridan et al. (2016) "Extreme Gradient Boosting as a Method for QSAR"
    JCIM — found max_depth=4, lr=0.1 needed to match RF on ECFP features
  - AutoGluon tabular XGBoost search space (n_estimators=10000, lr 0.005-0.2,
    max_depth 3-10, colsample_bytree 0.5-1.0, min_child_weight 1-5)
  - FLAML default (Wang et al. 2021, arXiv:2005.01571): meta-learned portfolio
    of XGBoost configs selected by 1-NN in (n_samples, n_features, n_classes,
    %numeric) space.  Key observations from the portfolio JSON files: all
    configs use lr 0.001-0.04 (far below XGBoost's default 0.3); the
    leaf-wise "xgboost" variant uses grow_policy="lossguide" + max_leaves
    (not max_depth) to build asymmetric trees; colsample_bylevel appears in
    every config; for high-p datasets the Amazon benchmark config
    (p≈10k) selects lr=0.001, colsample_bylevel=1.0, colsample_bytree=0.45.
  - Community heuristics from Kaggle and AnalyticsVidhya

No search, no cross-validation. Parameters are chosen purely from dataset
statistics captured in a DatasetProfile.

Key design decisions
--------------------
* early_stopping_rounds scales with 1/learning_rate so slower learners have
  enough patience to reach their optimum (50 rounds at lr=0.1; 250 at lr=0.02).
* The n/p ratio (samples per feature) drives max_depth and regularization:
  underdetermined problems (n/p < 5) need shallower trees and stronger L1/L2
  to avoid fitting noise.
* gamma (min_split_loss) scales with dataset difficulty: 0.05 for
  underdetermined dense data, 0.1 for underdetermined sparse-count data.
  Probst et al. 2023 ranks gamma #3 in importance for QSAR.  The previous
  0.5 for sparse-count data was too aggressive: in lossguide growth mode it
  pruned most splits before the tree reached max_leaves, leaving tiny trees
  that could not compete with default RF.  0.1 prunes only truly near-zero
  splits while allowing informative fingerprint patterns through.
* min_child_weight is raised for sparse-count data: requiring 3-10 samples per
  leaf effectively filters structural fragments that appear in fewer than that
  many training molecules, preventing the model from overfitting rare ECFP bits.
  Probst et al. 2023 recommends exploring 5-20 for fingerprint QSAR.
* learning_rate for sparse-count small datasets is 0.05 (midpoint of Probst
  et al. 2023's recommended 0.01-0.05 range; Sheridan et al. 2016 found
  lr=0.1 needed to match RF on ECFP features).  0.02 (previous value) was
  too conservative: combined with gamma pruning it caused underfitting vs
  default Random Forest.
* For sparse-count fingerprint data (ECFP/Morgan, p > 200), we use
  grow_policy="lossguide" + max_leaves instead of max_depth.  Leaf-wise
  growth (FLAML's "xgboost" portfolio variant) builds asymmetric trees that
  descend deeper in the few discriminative bit paths while not wasting
  capacity on the ~93% zero-valued bits.  max_leaves = max(16, min(128, n//50))
  gives 16-128 leaves depending on dataset size, providing similar capacity to
  a depth-4 symmetric tree (16 leaves) while allowing depth-unlimited
  specialisation where the signal is.
* For non-fingerprint data, depthwise growth is retained.  Binary-only features
  that are NOT sparse fingerprint data (pure one-hot style) get one depth level
  reduced, since each binary split carries roughly half the information of a
  continuous split.  The n/p ratio further caps depth for underdetermined data.
* Sparse count data (Morgan fingerprints etc.) uses max_bin=64; integer values
  ≤ 10 are perfectly captured by 64 histogram bins, halving memory versus 128.
* A small random forest component (num_parallel_tree=3) is injected for
  datasets with n < 2000 to reduce variance via within-round bagging —
  following XGBoost's own RF tutorial and practitioner guidance.  The range
  extends to n < 2000 (from the previous n < 1000) because drug datasets in
  the 1000-2000 range are still small enough to benefit from the variance
  reduction, and subsample=0.8 ensures per-tree diversity.
* colsample_bynode (per-split column sampling) is set for all high-dimensional
  data (p > 200), mirroring Random Forest's sqrt(p) per-split diversity.
  This applies equally to sparse fingerprints, dense physchem descriptors,
  and continuous embeddings: any high-p input benefits from per-node column
  diversity to prevent a handful of correlated features from dominating every
  split.  The formula sqrt(p)/p gives the RF-equivalent fraction; we clamp
  to [0.05, 0.3].  For sparse-count data colsample_bytree=1.0 ensures all
  features are eligible; for dense data the formula compensates for the
  bytree factor so the effective count targets sqrt(p).
* colsample_bytree for ECFP/sparse data is kept >= 0.6 so that the tree-level
  sampling does not additionally starve the per-node sampling budget.
* L1 regularization (reg_alpha) is beneficial for any underdetermined high-
  dimensional problem, not only ECFP fingerprints.  It drives uninformative
  leaf weights exactly to zero, performing implicit feature selection within
  each tree.  For sparse-count data the effect is strongest (many near-zero
  bits), so the L1 prior is heavier there.  For dense high-dim data
  (embeddings, physchem descriptors) with n/p < 5, a weak L1 (0.02-0.05)
  is added to the L2 baseline to help with redundant dimensions without
  distorting the overall gradient landscape.  Probst et al. 2023 confirms
  L1 is effective for QSAR tasks in general.
* feature_signal_strength (mean |Pearson| between features and target) adjusts
  the final regularization: very weak signal (< 0.02) tightens L1/L2; strong
  signal (> 0.05) eases L2 so the model can exploit available structure.
* n_estimators is set to 2000 (increased from 1000) with a correspondingly
  wider early-stopping window so that slow learners on small datasets can
  always reach their optimum before the ceiling is hit.
"""

import os
from typing import Dict, Any

from .inspector import DatasetProfile


def get_params(
    profile: DatasetProfile, device: str = "cpu", nthread: int = -1
) -> Dict[str, Any]:
    """
    Return a dict of XGBoost parameters for the given dataset profile.

    Parameters
    ----------
    profile : DatasetProfile
        Output of inspect(X, y).
    device : str
        "cpu" or "gpu".
    nthread : int
        Number of parallel threads.  -1 means "use all available CPU cores"
        (XGBoost default; ignored on GPU).

    Returns
    -------
    dict
        Ready to unpack into xgb.XGBClassifier(**params) or XGBRegressor(**params).
        Includes early_stopping_rounds; the caller must supply an eval_set when
        calling .fit().
    """
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")

    n = profile.n_samples
    p = profile.n_features
    params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Tree method and device
    # ------------------------------------------------------------------
    params["tree_method"] = "hist"
    params["device"] = "cuda" if device == "gpu" else "cpu"

    # ------------------------------------------------------------------
    # Learning rate
    # Slower learning on larger datasets: more samples → each tree is
    # already informative, so large steps overshoot.
    # For sparse-count (fingerprint) data on small datasets, Probst et al.
    # 2023 found lr 0.01-0.05 consistently outperforms 0.1: noisy bit-level
    # features benefit from smaller steps so the model doesn't commit to
    # spurious early trees.
    # ------------------------------------------------------------------
    if n < 10_000:
        if profile.is_sparse_counts:
            # Sheridan et al. 2016 found lr=0.1 was needed to match RF on
            # ECFP4 fingerprints.  For n<1000 (hERG, HIA_Hou-sized datasets)
            # 0.1 gives enough step size to converge within the early-stopping
            # budget; for 1000≤n<10000 (BBB-sized) 0.05 is sufficient.
            params["learning_rate"] = 0.1 if n < 1_000 else 0.05
        else:
            params["learning_rate"] = 0.1
    elif n < 100_000:
        # Sparse-count (fingerprint) data: lr=0.1 is sufficient at large n
        # (Sheridan et al. 2016); halves early_stopping_rounds (100→50) and
        # reduces estimated rounds ~3× vs lr=0.05.  Dense data keeps 0.05
        # for more careful convergence on continuous features.
        params["learning_rate"] = 0.1 if profile.is_sparse_counts else 0.05
    elif profile.is_sparse_counts:
        # Sparse fingerprint data at any large scale: lr=0.1 keeps cost
        # within budget (ratio ~6× default) and converges well regardless of n.
        # Gradient estimates on sparse integer bits are reliable even at n≥1M.
        params["learning_rate"] = 0.1
    elif n < 1_000_000:
        # Dense data, 100k–1M: careful lr=0.02 for continuous features where
        # gradient estimates improve more slowly with n.
        params["learning_rate"] = 0.02
    else:
        # Dense data at n≥1M: gradient estimates are very accurate; lr=0.05
        # converges ~5× faster than 0.02 at negligible quality cost.
        params["learning_rate"] = 0.05

    # ------------------------------------------------------------------
    # n_estimators and early stopping
    # Set a high ceiling; early stopping will find the optimal round.
    # early_stopping_rounds scales with 1/lr: slower learners need more
    # patience before improvement plateaus.  Formula:
    #   patience ≈ 50 × (0.1 / lr)
    # giving 50 rounds at lr=0.1, 100 at lr=0.05, 250 at lr=0.02.
    # Ceiling raised to 2000 so slow learners on small fingerprint
    # datasets (lr=0.1, ~50 patience) still have room to converge.
    # ------------------------------------------------------------------
    params["n_estimators"] = 2000
    params["early_stopping_rounds"] = max(
        20, int(round(50 * (0.1 / params["learning_rate"])))
    )

    # ------------------------------------------------------------------
    # Tree growth strategy
    #
    # Sparse-count fingerprint data (ECFP/Morgan, p > 200):
    #   grow_policy="lossguide" + max_leaves  (FLAML "xgboost" variant)
    #   Leaf-wise growth builds asymmetric trees that can go arbitrarily
    #   deep along the few discriminative bit paths while ignoring the
    #   ~93% zero-valued bits.  max_depth=0 disables the depth cap;
    #   max_leaves controls total capacity instead.
    #   max_leaves = max(16, min(128, n // 50)) gives:
    #     n=500  → 16 leaves (conservative for very small datasets)
    #     n=1600 → 32 leaves (~equivalent to a balanced depth-5 tree)
    #     n=5000 → 100 leaves
    #     n≥6400 → 128 leaves (ceiling)
    #
    # All other data: standard depthwise growth.
    #   Larger datasets support deeper trees; n/p ratio and binary-feature
    #   fraction further modulate depth.
    # ------------------------------------------------------------------
    if profile.is_sparse_counts and p > 200:
        params["grow_policy"] = "lossguide"
        params["max_depth"] = 0  # unlimited; max_leaves takes over
        # max_leaves raised to give enough tree capacity on small datasets.
        # Old formula (n//50) gave only 16 leaves for n<800, capping the
        # model at ~36 samples/leaf — too coarse for underdetermined data.
        # New formula (n//10) gives 64 leaves minimum (≈depth-6 tree) so
        # the model can express meaningful structure before regularization
        # takes over.  Formula:
        #   n=578  → 64 (floor at 64)
        #   n=1624 → 162
        #   n=7278 → 512 (ceiling)
        # Cap at 256: generous capacity (4× default's 64-leaf depth-6 tree)
        # while halving per-round cost vs the previous 512 cap.
        params["max_leaves"] = max(64, min(256, n // 10))
    else:
        if n < 1_000:
            max_depth = 3
        elif n < 10_000:
            max_depth = 4
        elif n < 100_000:
            max_depth = 5
        else:
            max_depth = 6

        # n/p ratio cap: underdetermined problems overfit with deep trees.
        if profile.n_p_ratio < 2:
            max_depth = min(max_depth, 3)
        elif profile.n_p_ratio < 5:
            max_depth = min(max_depth, 4)

        # Binary-feature-rich inputs (one-hot style): reduce depth by one.
        if profile.binary_feature_fraction > 0.8:
            max_depth = max(3, max_depth - 1)

        params["max_depth"] = max_depth

    # ------------------------------------------------------------------
    # min_child_weight
    # Scales with dataset size to prevent leaves supported by very few
    # samples. Halved for severely imbalanced classification so the
    # minority class can form leaves at all.
    # For sparse-count (fingerprint) data, Probst et al. 2023 recommends
    # 5-20: requiring N molecules to share a structural fragment before it
    # becomes a split criterion filters rare ECFP bits that correlate with
    # the target by chance.  We use max(3, n//500) which gives 3-10 in
    # the typical drug dataset range (1000-5000 molecules).
    # ------------------------------------------------------------------
    if profile.is_sparse_counts:
        mcw = max(3, n // 500)
    else:
        mcw = max(1, n // 1000)
    mcw = min(mcw, 20)

    if profile.task == "classification":
        if profile.imbalance_ratio > 10:
            # Extreme minority positives: halve mcw so rare positives can
            # form leaves at all.
            mcw = max(1, mcw // 2)
        elif profile.imbalance_ratio < 1:
            # Positives are the MAJORITY class.  XGBoost's min_child_weight
            # is based on the sum of hessians, not sample count.  When
            # scale_pos_weight = imbalance_ratio < 1, each positive sample's
            # hessian is scaled down by that factor, so a leaf needs
            # mcw / (scale_pos_weight × 0.25) ≈ mcw / (ratio × 0.25)
            # positives to meet the threshold — which can be most of the
            # training set for small ratio values (e.g. HIA_Hou: ratio=0.16
            # → needs 75 out of 500 positives per leaf, preventing splits).
            # Scale mcw down proportionally so majority-class leaves can form.
            mcw = max(1, round(mcw * profile.imbalance_ratio))

    params["min_child_weight"] = mcw

    # ------------------------------------------------------------------
    # subsample
    # Stochastic gradient boosting: Friedman showed that subsample in
    # [0.3, 0.8] almost universally helps.  For tiny datasets subsampling
    # introduces more variance than it removes — use all rows instead.
    # Reduce further for very large datasets to save memory.
    # ------------------------------------------------------------------
    # For small datasets (n < 1000), every sample carries meaningful gradient
    # signal; row subsampling introduces variance that outweighs any benefit.
    # Default XGBoost (subsample=1.0) outperforms us on HIA_Hou/hERG precisely
    # because it uses all rows each round.  Keep Friedman's 0.8 for n ≥ 1000
    # where stochastic boosting reduces variance without starving individual trees.
    if n < 1_000:
        params["subsample"] = 1.0
    elif n >= 1_000_000:
        params["subsample"] = 0.6
    else:
        params["subsample"] = 0.8

    # ------------------------------------------------------------------
    # num_parallel_tree  (random-forest style within-round bagging)
    # For small datasets, training a small forest at each boosting step
    # reduces variance via bagging diversity without cross-validation.
    # subsample < 1 (0.8, set above) ensures the parallel trees see
    # different row subsets within each round.
    # Range extended to n < 2000 because drug datasets in the 1000-2000
    # range are still underdetermined relative to fingerprint dimensionality
    # (n/p ratio typically 1-2) and benefit from the additional variance
    # reduction.  num_parallel_tree=3 follows the XGBoost RF tutorial
    # recommendation and Kaggle practitioner consensus for boosted RF.
    # ------------------------------------------------------------------
    # num_parallel_tree is only beneficial when there are enough samples per
    # tree to give stable gradients.  For n/p < 0.8 (severely underdetermined),
    # each of the 3 parallel trees would see ≤ 80% of an already tiny dataset,
    # producing noisy estimates.  Plain boosting with all rows (subsample=1.0,
    # set above for n<1000) is more sample-efficient in this regime.
    if 200 <= n < 2_000 and profile.n_p_ratio >= 0.8:
        params["num_parallel_tree"] = 3

    # ------------------------------------------------------------------
    # colsample_bytree
    # Scales down with feature count to reduce variance and memory.
    # For sparse-count fingerprint data (is_sparse_counts), colsample_bytree
    # is set to 1.0 so that colsample_bynode can sample independently from
    # ALL p features at every split — exactly mirroring Random Forest.
    # If bytree < 1, a fixed subset of features is excluded from all splits
    # in a tree, breaking the "any feature can appear at any split" guarantee
    # that makes RF effective on high-dimensional binary data.
    # For other data, tree-level subsampling reduces variance and memory.
    # ------------------------------------------------------------------
    if profile.is_sparse_counts:
        cst = 1.0
    elif p <= 50:
        cst = 1.0
    elif p <= 200:
        cst = 0.8
    elif p <= 500:
        cst = 0.7
    elif p <= 2000:
        cst = 0.5
    else:
        cst = max(0.3, 500 / p)

    params["colsample_bytree"] = round(cst, 2)

    # ------------------------------------------------------------------
    # colsample_bynode  (per-split column sampling)
    # For any high-dimensional input (p > 200) — sparse fingerprints, dense
    # physchem descriptors, or continuous embeddings — sampling columns at
    # each split injects RF-style per-node diversity and prevents a few
    # correlated features from dominating every split.
    #
    # Target: effective features per split ≈ sqrt(p)  (RF standard).
    # For is_sparse_counts: colsample_bytree=1.0, so csn = 1/sqrt(p).
    # For dense data:        colsample_bytree<1, so csn = 1/(sqrt(p)×cst)
    #                        to compensate and still hit sqrt(p) effective.
    # Formula examples (sparse, colsample_bytree=1.0):
    #   p=512  → 0.044 → clamped to 0.05  → ~26 features (RF: 23)
    #   p=1024 → 0.031 → clamped to 0.05  → ~51 features (RF: 32)
    # Formula examples (dense, colsample_bytree=0.7, p=512):
    #   csn = 1/(22.6×0.7) ≈ 0.063 → ~23 features ≈ sqrt(512) ✓
    #
    # For sparse-count data with n/p < 1 (very underdetermined), the floor
    # is raised from 0.05 to 0.1 so key fingerprint bits have a higher
    # chance of being sampled at each split.  Dense data keeps floor=0.05
    # since continuous features carry more information per split.
    # ------------------------------------------------------------------
    if p > 200:
        if profile.is_sparse_counts:
            csn_floor = 0.1 if profile.n_p_ratio < 1.0 else 0.05
            csn = round(max(csn_floor, min(0.3, 1.0 / (p**0.5))), 3)
        else:
            # Dense: compensate for bytree so bytree × csn ≈ sqrt(p)/p.
            csn = round(max(0.05, min(0.3, 1.0 / (p**0.5 * cst))), 3)
        params["colsample_bynode"] = csn

    # Note: colsample_bylevel is intentionally NOT set when colsample_bynode
    # is active.  All three colsample parameters multiply together:
    #   effective_features = p * bytree * bylevel * bynode
    # colsample_bynode is already calibrated so that
    #   p * bytree * bynode ≈ sqrt(p)  (RF per-split target).
    # Adding bylevel=0.7 would push effective features to only 0.7×sqrt(p),
    # undershooting the RF target.  bylevel also has no clear meaning
    # in lossguide (leaf-wise) growth mode, which is not level-wise.

    # ------------------------------------------------------------------
    # Regularization
    #
    # Previous design used multiplicative stacking (base × n/p-multiplier
    # × n<1000-multiplier × signal-multiplier) which silently compounded
    # to extremes (e.g. lambda=10 for n=500, p=1024).  Replaced with a
    # direct table keyed on (is_sparse_counts, n/p) so each regime has
    # explicit, auditable values that can't compound uncontrollably.
    #
    # Design rationale:
    # - L1 (reg_alpha) drives uninformative ECFP leaf weights to exactly
    #   zero, performing implicit feature selection.  Critical for sparse
    #   fingerprint data where ~93% of bits are near-zero.  Probst et al.
    #   2023 confirms L1 is effective for QSAR.
    # - L2 (reg_lambda) smooths leaf weights.  Keeps at 1.0 for well-
    #   determined data (XGBoost default), raised for underdetermined.
    # - n<1000 adds a flat +0.5 to lambda (not a multiplier) to account
    #   for the additional variance from very small training sets without
    #   the explosive stacking of previous multiplicative design.
    # - Signal-strength modulates ±15%: the adjustment is modest because
    #   the table values are already calibrated for each regime.
    # ------------------------------------------------------------------
    if profile.is_sparse_counts:
        if profile.n_p_ratio < 1:
            # Very underdetermined (n < p): light regularization so the
            # model can still learn — heavy penalties collapse leaf weights
            # toward zero when each tree has already very few samples.
            params["reg_alpha"] = 0.3
            params["reg_lambda"] = 2.0
        elif profile.n_p_ratio < 2:
            # Underdetermined (n/p 1–2): moderate L1, modest L2 increase.
            params["reg_alpha"] = 0.5
            params["reg_lambda"] = 1.5
        elif profile.n_p_ratio < 5:
            # Mildly underdetermined: stronger L1 for bit-level selection.
            params["reg_alpha"] = 1.0
            params["reg_lambda"] = 1.0
        else:
            # Well-determined fingerprint data: light L1 sufficient.
            params["reg_alpha"] = 0.1
            params["reg_lambda"] = 1.0
    else:
        # Dense continuous features: primarily L2, with weak L1 for
        # high-dim underdetermined data (embeddings, physchem with p>100).
        # L1 drives redundant dimensions toward zero; the values are
        # deliberately light (0.02-0.05) so the gradient landscape is
        # not distorted — unlike the stronger L1 used for sparse bits.
        if profile.n_p_ratio < 2:
            params["reg_alpha"] = 0.05 if p > 100 else 0.0
            params["reg_lambda"] = 2.0
        elif profile.n_p_ratio < 5:
            params["reg_alpha"] = 0.02 if p > 100 else 0.0
            params["reg_lambda"] = 1.5
        else:
            params["reg_alpha"] = 0.0
            params["reg_lambda"] = 1.0

    # Flat small-dataset bonus: +0.5 lambda for n<1000.
    if n < 1_000:
        params["reg_lambda"] = params["reg_lambda"] + 0.5
        params["reg_alpha"] = max(params["reg_alpha"], 0.3)

    # Signal-strength modulation (±15%).
    # feature_signal_strength (mean |Pearson|) can be misleading for
    # fingerprint data: the mean is dragged down by ~93% uninformative bits
    # even when the top bits are highly predictive.  Use feature_signal_p90
    # (90th-percentile |Pearson|) to detect whether the best features are
    # informative, even when the average is not.
    sig = profile.feature_signal_strength
    sig_p90 = profile.feature_signal_p90
    if sig < 0.02 and sig_p90 < 0.05:
        # Genuinely near-random data: tighten slightly.
        params["reg_lambda"] = round(params["reg_lambda"] * 1.15, 4)
        params["reg_alpha"] = round(params["reg_alpha"] * 1.15, 4)
    elif sig > 0.05 or sig_p90 > 0.10:
        # Real signal present: ease lambda so the model can exploit it.
        params["reg_lambda"] = round(params["reg_lambda"] * 0.85, 4)

    # Hard caps: beyond these values the model collapses leaf weights to
    # near-zero and cannot differentiate between samples.
    params["reg_lambda"] = round(min(params["reg_lambda"], 4.0), 4)
    params["reg_alpha"] = round(min(params["reg_alpha"], 1.5), 4)

    # ------------------------------------------------------------------
    # gamma (min_split_loss)
    # Pre-pruning regularizer: a split is accepted only if it reduces the
    # loss by at least gamma.  Probst et al. 2023 ranks this #3 in
    # importance for QSAR.  Rules are uniform across feature types:
    # - n/p < 1 (very underdetermined): gamma=0.05 — light pruning so the
    #   model can still find the few valid splits available.
    # - 1 ≤ n/p < 5 (mildly underdetermined): gamma=0.05 — prunes only
    #   near-zero-gain splits.  Applies to all data types uniformly.
    # - n/p ≥ 5 (well-determined): gamma=0 — L2/L1 regularization suffices.
    if profile.n_p_ratio < 1:
        params["gamma"] = 0.05
    elif profile.n_p_ratio < 5:
        # Uniform 0.05 for all underdetermined data: prunes only near-zero-gain
        # splits regardless of feature type (fingerprints, physchem, embeddings).
        params["gamma"] = 0.05

    # ------------------------------------------------------------------
    # max_bin (histogram granularity)
    # Sparse count features (integer values ≤ 10) are perfectly captured
    # by 64 bins — using more is wasteful and slows histogram construction.
    # For large or high-dimensional dense data, 128 balances accuracy and
    # memory.  The default 256 is used otherwise.
    # ------------------------------------------------------------------
    if profile.is_sparse_counts:
        params["max_bin"] = 64
    elif n > 1_000_000 or p > 500:
        params["max_bin"] = 128
    else:
        params["max_bin"] = 256

    # ------------------------------------------------------------------
    # Parallelism (CPU only; GPU manages its own threads)
    # nthread=-1 → use all available CPU cores (XGBoost default behaviour).
    # A user-supplied nthread > 0 overrides this for reproducibility or
    # resource-constrained environments.
    # ------------------------------------------------------------------
    if device == "cpu":
        params["nthread"] = (os.cpu_count() or 1) if nthread == -1 else nthread

    # ------------------------------------------------------------------
    # Task-specific parameters
    # ------------------------------------------------------------------
    if profile.task == "classification":
        _set_classification_params(params, profile)
    else:
        _set_regression_params(params, profile)

    return params


def _set_classification_params(params: Dict[str, Any], profile: DatasetProfile) -> None:
    params["objective"] = "binary:logistic"

    ratio = profile.imbalance_ratio

    # scale_pos_weight: equivalent to sklearn's class_weight="balanced".
    # Always set it — when ratio=1 it is a no-op, and for any imbalance
    # (even mild) it corrects the gradient contribution of each class.
    params["scale_pos_weight"] = round(ratio, 4)

    # max_delta_step = 1 stabilises logistic regression gradient updates for
    # imbalanced data (XGBoost docs recommendation).  Threshold lowered from
    # 100 to 10: a 10:1 imbalance already produces biased gradient scales
    # that benefit from clamping, and datasets like ChEMBL (67:1) were
    # previously missing this stabilisation.
    if ratio > 10:
        params["max_delta_step"] = 1

    # AUC-PR is a more informative metric than AUC-ROC for imbalanced data
    # because it focuses on the minority (positive) class.
    if ratio > 10:
        params["eval_metric"] = "aucpr"
    else:
        params["eval_metric"] = "auc"


def _set_regression_params(params: Dict[str, Any], profile: DatasetProfile) -> None:
    skew = profile.y_skewness
    abs_skew = abs(skew)

    if abs_skew < 1.0:
        # Approximately symmetric: standard MSE loss is appropriate.
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse"

    elif abs_skew < 2.0:
        # Moderate skew with positive y: Tweedie loss handles right-skewed,
        # non-negative distributions (e.g. count data, insurance losses).
        # For negative or mixed-sign y fall back to pseudoHuber which is
        # robust to outliers without requiring positivity.
        if profile.y_all_positive:
            params["objective"] = "reg:tweedie"
            params["tweedie_variance_power"] = 1.5
            params["eval_metric"] = "tweedie-nloglik@1.5"
        else:
            params["objective"] = "reg:pseudohubererror"
            params["eval_metric"] = "mae"

    else:
        # Severe skew: pseudoHuber is robust to the heavy tail regardless
        # of sign. For positive-only targets Tweedie is also valid but
        # pseudoHuber makes no distributional assumptions.
        if profile.y_all_positive:
            params["objective"] = "reg:tweedie"
            params["tweedie_variance_power"] = 1.5
            params["eval_metric"] = "tweedie-nloglik@1.5"
        else:
            params["objective"] = "reg:pseudohubererror"
            params["eval_metric"] = "mae"
