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
- **`rank` is now the training-set percentile of the pooled probability.** It was a
  weighted mean of per-descriptor percentiles, which is not a monotone transform of the
  pooled probability — so the same model could order two molecules one way by `proba` and
  the other way by `rank`. It also made `rank` the output least able to survive the ONNX
  export: an ECDF is steep wherever the training scores bunch up, so it amplified a
  difference `proba` barely registered. Measured on the ChEMBL fixtures, `proba` agreed
  with its export to 9e-08 while `rank` moved by up to 0.09 on 500 of 665 molecules
  (Spearman 0.9926). `rank` is now that pooled probability read off a single pooled
  out-of-fold reference stored in the checkpoint, which makes it a monotone view of
  `proba`: Spearman 1.0000000 against the export, and zero churn in the top 1%, 5% and
  10%. Two consequences worth planning for — rank-based AUROC, AUPRC and BEDROC now equal
  the probability-based ones, and the values spread over a wider range than the averaging
  produced, so an absolute cutoff such as `rank > 0.8` selects a different set than
  before. Checkpoints fitted by earlier versions carry no reference and keep their
  previous behaviour exactly; refit to get the new one.
- **`score` is unchanged on checkpoints without an applicability domain** (2e-08). With
  uniform weights a weighted mean is the arithmetic mean, so it was already correct.
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

- **The exported ONNX checkpoint now matches the model it was exported from.** It did not:
  the exported preprocessor ran in float32 while scikit-learn ran it in float64, and
  although the two agreed to about one float32 ULP, the tree heads downstream are
  piecewise constant — a value landing a hair either side of a learned split fell into a
  different leaf and moved the score by ~0.1. The fitted preprocessor now runs its own
  exported graph, so the heads are fitted on bit-identical values to the ones they are
  later served. Measured on the non-separable fixture that pinned the defect, agreement
  went from ~1e-01 to 5.7e-08 in `score`; out-of-fold AUC is unchanged to four decimal
  places on both ChEMBL fixtures and on that fixture, so this costs no accuracy. The
  `xfail(strict=True)` on `test_exported_model_matches_the_model_it_came_from` is gone.
- The fitted model in memory weighted its descriptors by OOF AUC while every loader
  weighted them by `quality_aucs` (`2 * oof - train`), so the same model predicted
  slightly differently before and after `save()` / `load()`. All paths now use quality.
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

- The task-level `decision_cutoff_rank` is now the learned probability cutoff expressed
  against the pooled reference, rather than a mean of the per-descriptor rank cutoffs,
  which after the change sat on a scale nothing emits. It is reported only — nothing in
  the package thresholds on it, and `binary` is still `proba >= 0.5`.
- `ArtifactWrapper` streams inference in chunks instead of featurizing the whole input at
  once. Scoring a million compounds against a 2048-dimensional descriptor no longer needs
  ~8 GB before inference starts. `LAZYQSAR_PREDICT_CHUNK` sets the batch size (default 1000).
- Every requested output now comes from a single featurization pass. Asking for `proba` and
  `rank` used to featurize twice, and featurization is ~93% of inference wall clock.
- `predict()` accepts both `threshold=` and `cutoff=` for the binary threshold.
- **Prediction is bounded in memory.** Molecules are scored in blocks whose working set is
  released before the next block starts, so peak memory no longer grows with the size of
  the input. It used to: the accumulated per-task channels and the combined results were
  held for the whole call, which at a million molecules and twenty endpoints came to about
  3.6 GB, and `LAZYQSAR_PREDICT_CHUNK` did nothing about it — it bounds the descriptor
  slice, not the accumulation. Peak temporary disk drops with it, from 8.2 GB to under 1 GB
  at that size. The block is derived from a fixed working-set budget and the number of
  endpoints, and is always a whole number of chunks, which is what keeps the output
  identical however the input is divided; there is no knob for it.
- **Asking for `rank` no longer doubles the ONNX work.** On a checkpoint carrying the
  pooled reference a rank is that reference evaluated at the pooled probability — which the
  scoring loop has already computed for the same rows. It was calling `predict_rank`, which
  re-ran every preprocessor and every head to reach the identical number. Measured at 10
  ONNX calls per chunk against 5. This is the default the Ersilia template deploys with,
  and a checkpoint with an applicability domain needs the rank channel even for a plain
  `proba` request, so it was most of the cost of the two commonest requests.
- **Descriptors the portfolio rejected are no longer loaded.** Both Python loaders opened
  every descriptor directory on disk — featurizer, ONNX sessions, applicability domain —
  and then scored only the active ones, which they had always done. On a five-descriptor
  checkpoint where two survived, 25 ONNX sessions became 10.
- Molecules RDKit cannot parse are found by re-checking only the rows a descriptor already
  returned as all-NaN, instead of parsing the whole library a second time after every
  descriptor has already parsed it. A clean library now costs no parses at all. Descriptors
  that repair NaN rows — `cddd` fetches a ChEMBL nearest neighbour — are excluded from the
  shortcut, and a run using only those falls back to the full scan.
- Morgan fingerprints are written into a preallocated array rather than a list of lists:
  46 ms against 0.6 ms per thousand molecules, roughly 45 seconds per million on top of
  RDKit's own cost. Same values, same dtype.
- The CheMeleon forward pass runs under `torch.no_grad()`. `eval()` was called but autograd
  was still recording a graph for every batch, for a model that is never trained here.
- `combine()` no longer allocates a full `(n_samples, n_descriptors)` float64 copy of the
  probabilities on every call. It was the fallback for a missing raw score, and nothing
  reads it unless `score` is among the requested outputs.
- **The fit-time descriptor union has a reproducible row order.** It was built from
  `os.listdir` and `set`, so the same command over the same data laid the staged descriptor
  matrices out differently from one process to the next. Nothing read the wrong row, but a
  fit could not be reproduced exactly. Filenames are now sorted and duplicates dropped in
  first-seen order.

### Internal

- New `lazyqsar/ensemble/`: `combine()` (the weighting and the six output formulas, pure
  numpy), `channels.py` (chunked scoring) and `runner.py` (multi-task inference). Four
  near-duplicate implementations became one.
- New `lazyqsar/registry.py` and `lazyqsar/applicability/artifact.py`, both so the
  inference path imports without RDKit or scikit-learn.
- `persist_descriptors` returns the all-NaN row mask it can see for free while each chunk
  is still in cache, and `predict_tasks` takes an optional `scan` dict reporting those rows
  and the descriptors that ran. An out-parameter rather than a changed return type, so
  existing callers are untouched; only the CSV-writing layer wants it.
- No ONNX converter, opset or graph was changed, and nothing under `lazyqsar/base/` was
  modified beyond one import. `lazyqsar/artifacts/classifier.py` gained one method,
  `rank_from_proba`, which reads a rank off a probability the caller already has; it runs
  no graph and the values are identical to what `predict_rank` returns.
