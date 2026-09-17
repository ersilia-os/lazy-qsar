# Changelog

## 3.5.0

The CLI and the Python API now share one implementation. Before this release they fitted
and combined models differently, so the same molecule could get a different answer
depending on which entry point produced and scored it — by up to 0.22 in probability.
After it, the four fit × predict combinations agree to float32 noise (1.7e-08).

Accuracy is essentially unchanged. On a held-out antimicrobial benchmark AUROC moved
0.958 → 0.962; every entry point converged on what the Python path already achieved.
This release removes an inconsistency, it does not claim a better model.

### Breaking

- **`binary` is now a 0/1 label.** The CLI averaged per-descriptor labels, so a
  five-descriptor model emitted `{0, 0.2, 0.4, 0.6, 0.8, 1.0}`. It is now one label taken
  from the pooled probability, thresholded at 0.5 — what the Python API always did.
- **`proba`, `logit` and `lift` change on existing checkpoints.** Descriptors are combined
  in logit space rather than by averaging probabilities. Measured on a checkpoint without
  an applicability domain: `proba` moves up to 0.05, Spearman 0.9989. With one: up to
  0.13, Spearman 0.9929.
- **`rank` and `score` are unchanged on checkpoints without an applicability domain**
  (2e-08). With uniform weights a weighted mean is the arithmetic mean, so these were
  already correct.
- **New CLI checkpoints differ from old ones.** `lazyqsar fit` now applies the descriptor
  portfolio and fits an applicability domain, so it keeps 2–3 descriptors instead of all
  5 and writes `applicability_domain/` per descriptor. Predictions differ from checkpoints
  built by earlier versions, and inference is correspondingly faster. Refitting is what
  makes the two entry points agree — upgrading the code alone does not, because a 5-model
  checkpoint and a 3-model checkpoint are different models.
- **`LazyClassifierQSAR.load()` returns `ArtifactWrapper`** for ONNX checkpoints, as its
  documentation always claimed. It previously fell through to the raw loader for every
  checkpoint.

### Fixed

- The inference path could not be imported on a base install: it reached `scipy` through
  a chain into fit-time code, and `scipy` is in the `fit` extra. Any environment installed
  as documented for inference failed at import.
- `LazyClassifierQSAR.load()` looked for ONNX graphs one directory too shallow, so it never
  found them. `ArtifactWrapper` and `load_onnx` were unreachable.
- The two loaders read different statistics for the descriptor weighting — `quality_aucs`
  and `oof_aucs` — so the same checkpoint was weighted differently depending on which one
  ran. Both now use `quality_aucs`, falling back to `oof_aucs`.
- `predict()` given two column names pointing at the same directory silently dropped one.
- `api.classifier_fit.fit()` defaulted to `mode="default"`, which is not a valid mode and
  raised `KeyError`. It now defaults to `slow` and validates.
- The inference-purity test hooked `find_module`, removed as an import hook in Python 3.12,
  so it had been passing regardless of what was imported.
- Fit-time scratch files were written into the user's output directory and left behind on
  a crash.

### Changed

- `ArtifactWrapper` streams inference in chunks instead of featurizing the whole input at
  once. Scoring a million compounds against a 2048-dimensional descriptor no longer needs
  ~8 GB before inference starts. `LAZYQSAR_PREDICT_CHUNK` sets the batch size (default 1000).
- Every requested output now comes from a single featurization pass. Asking for `proba` and
  `rank` used to featurize twice, and featurization is ~93% of inference wall clock.
- `predict()` accepts both `threshold=` and `cutoff=` for the binary threshold.

### Internal

- New `lazyqsar/ensemble/`: `combine()` (the weighting and the six output formulas, pure
  numpy), `channels.py` (chunked scoring) and `runner.py` (multi-task inference). Four
  near-duplicate implementations became one.
- New `lazyqsar/registry.py` and `lazyqsar/applicability/artifact.py`, both so the
  inference path imports without RDKit or scikit-learn.
- No ONNX converter, opset or graph was changed, and nothing under `lazyqsar/artifacts/`
  or `lazyqsar/base/` was modified beyond one import.
