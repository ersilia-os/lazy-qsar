# Changelog

## 3.6.0

`predict_rank` now positions a molecule against a fixed reference library of 50,000
drug-like molecules, anchored on that library's **upper tail**: 0.50 means the top 10% of
drug-like chemical space, 0.65 the top 1%, 0.75 the top 0.1%. Each step is a 10x shrink of
the tail. Before this release it was a percentile against the model's own training set,
which sounds similar and is not.

`binary` and the decision cutoff follow from that scale. The cutoff is fixed at **rank
0.65** -- "to be called a hit, beat 99% of drug-like space" -- so a model calls 1% of
generic chemistry active by construction, and `proba`, `logit`, `lift` and `score` cutoffs
are all derived from that one probability.

The old behaviour was not a bug in the ranking arithmetic. A trained model's out-of-fold
scores are bimodal -- inactives crushed near 0, actives near 1, almost nothing between --
so real screening compounds land in the empty middle where the training ECDF is flat.
Measured on a simulated screening library, the entire top decile spanned 0.0014 of rank
and 53.6% of the library compressed into [0.75, 0.95]: everything came back at roughly 0.9
and users could not tell their best compounds from their mediocre ones. The training ECDF
was correct; the training set simply contained nothing in that score range, so no change
to the interpolation could manufacture resolution there. Two cheaper fixes were measured
and rejected -- correcting tie handling moved the mean from 0.670 to 0.669, and ranking
against training negatives made it worse at 0.748.

### Why the tail and not the quartiles

Quartile anchors spend three quarters of the axis below the library's median, on compounds
nobody will order. A bioactivity model's product is the top of the list, and measured across
six real antimicrobial models the top 1% of a screened library sits between the reference's
p98.9 and p100 -- exactly where the quartile scale had no resolution left. The rank-axis span
given to the top 1% of DrugBank went from a mean of 0.069 to 0.215; on one model the 113
best-scoring compounds had shared a span of 0.003, effectively a single value, and now have
0.096.

The cost is taken deliberately: **the bottom of the scale is flattened.** Roughly a third of
a generic library can land below rank 0.25. That is the right trade for a hit list, but it
means `rank` carries little information about *how* inactive something is, and nothing
downstream should read low ranks quantitatively.

An earlier measurement scored the scales by how *evenly* they spread a whole screening
library, and on that basis the tail anchors looked worst. That metric values resolving
inactives, which for a bioactivity model is worth nothing; scored on the top of the list
instead, the ordering reverses and the tail anchors win on every model at every cut.

### Why the cutoff is a constant

The cutoff used to be learned: `balanced_accuracy_score` maximised over out-of-fold raw
scores, whose argmax is Youden's J. That is optimal against the inactives *measured in the
same assay* -- usually close analogues from one programme -- and says nothing about generic
chemistry. Measured on six real models it gave generic hit rates of **0.60 to 0.99**: every
one called most of drug-like space active, and one called 98.3% of DrugBank a hit.

Deriving it from the rank scale instead needs no new data -- the reference library is already
scored at fit time to build the knots -- and 0.65 sits below the last reference anchor, so
inverting it never touches the out-of-fold actives anchor. The cutoff is therefore a pure
function of the reference library and does not wobble on a model fitted from 24 actives.

Note that maximising an *ordering-based* criterion on the rank scale would have changed
nothing: rank is a monotone transform of `proba`, so Youden's J picks the identical split
either way and only the printed number moves. A fixed point on the scale is the only version
of this that has any effect.

At fit time the model now scores the reference library and stores the sorted pooled
probabilities as its knots. Inference is unchanged: still one interpolation against knots
in `metadata.json`, so deployed models need no new data and inference still requires only
numpy and onnxruntime.

### Breaking

- **`rank` changes meaning, and old checkpoints refuse to report it.** Checkpoints fitted
  before 3.6 carry out-of-fold knots, which are indistinguishable from library knots once
  read -- both monotone, both in [0, 1]. Rather than report a training-set percentile as
  though it were a library percentile, `predict_rank` and `--predict_type rank` raise and
  name the fix. **`proba`, `logit`, `lift`, `score` and `binary` are unaffected** and keep
  working on existing checkpoints; refit to get `rank` back.
- **The pre-3.5 fallback is gone.** `combine` no longer falls back to the weighted mean of
  per-descriptor training percentiles. It answered a different question, and keeping it
  would have meant one `rank` column meaning two different things depending on when the
  checkpoint was fitted, with nothing in the output to say which.
- **`decision_cutoff_rank` is now exactly `0.65` on every model**, and `binary`
  thresholds on it. What moves is `decision_cutoff_proba`, which is whatever 0.65 inverts
  to for that model's reference. On the six antimicrobial models measured, the share of
  DrugBank called active drops from 68-98% to 0.9-3.6%.

  Gated on a new `decision_cutoff_source: "reference_rank"` key rather than on the presence
  of `decision_cutoff_proba`, because every checkpoint has one of those. A checkpoint
  without the key keeps `proba >= 0.5`, so **its labels are bit-identical to before.**
- **`decision_cutoff_raw` changes scale.** It was a mean of per-head raw cutoffs living on
  different quantities -- XGB raw probability, RF vote fraction, SVC sigmoid-of-margin, LR
  probability -- which corresponded to nothing any output emits. It is now the cutoff in the
  units the task's `score` output emits. Reporting only: `score_from_knots` is a running-max
  staircase, so `score >= decision_cutoff_raw` is **not** equivalent to `binary`.
- **`LazyClassifierQSAR.predict()` returns `combine`'s `binary`.** It used to recompute the
  label from `proba` against a hardcoded 0.5, ignoring the `binary` it had just computed, so
  the Python API and `--predict_type binary` could disagree. Pass `threshold=` to override.
- **`LazyClassifier.predict()` (agnostic) thresholds proba, not score.** With a reference it
  compares `predict_proba` against the rank-derived cutoff; without one it keeps comparing
  `predict_score` against the balanced-accuracy cutoff, unchanged. The two entry points
  agreed with neither each other nor the CLI before this.
- **`anchor_low` no longer shapes the scale.** The tail table has no low anchor, so
  `anchor_low_used` is always `False`. `p05` of the out-of-fold inactives is still measured
  and recorded, because where a model's inactives sit is worth reporting.
- **Fitting requires the reference library.** `lazyqsar setup --reference` fetches it, and
  only the descriptors a model actually uses are downloaded -- 6.5 MB for a `fast` model,
  267 MB for all five.

### Added

- **`sensitivity_at_cutoff` in `oof_diagnostics`** -- the share of a model's own known
  actives it still catches while calling 1% of drug-like chemical space active. Sensitivity
  at a fixed generic hit rate, the standard screening statistic.

  This carries the signal `generic_hit_rate` gives up: now that the cutoff is a fixed point
  on the rank scale, the hit rate is 0.01 by construction on every model. `generic_hit_rate`
  is kept anyway, because it is a one-line check that inverting the rank landed where it
  claims. Measured across six models `sensitivity_at_cutoff` reads 0.042 to 0.622, which
  ranks them sensibly; a model fitted on shuffled labels catches almost none.

  Precision at the cutoff is deliberately *not* reported. The out-of-fold inactives are
  assay-matched analogues rather than generic chemistry, so a precision here would look
  usable and not be -- the same trap the `oof_auc`-versus-`screening_auc` note describes.

- **`oof_diagnostics` in every checkpoint, and in the fit log.** A rank on its own cannot be
  judged, so a fitted model now records how it treats molecules whose labels are known:
  where its out-of-fold actives and inactives land on the rank scale (quartiles, with
  counts), a `screening_auc`, and a `generic_hit_rate`.

  `screening_auc` is the out-of-fold actives against the reference library, and it answers a
  question `oof_auc` does not. `oof_auc` separates actives from *measured inactives for that
  target*, usually close analogues from the same assay; a screen instead asks whether actives
  rise above generic chemical space. A model can do the first well and the second badly, and
  then a real screen drowns in false positives.

  `generic_hit_rate` is the share of drug-like chemical space the model would call active.
  On the ChEMBL fixture a working model reports 0.6%; the same model fitted on shuffled
  labels reports 34.9%, which is a more actionable statement about it than any accuracy
  metric.

  These are advisory and never enter the rank scale. Anchoring the scale on the out-of-fold
  actives -- so that "0.95 means looks like a known active" -- was considered and rejected,
  because it would put every model's median active at 0.95 by construction and make a model
  with AUC 0.95 indistinguishable from one with AUC 0.55. Reported rather than anchored, the
  numbers keep that signal: on shuffled labels the actives' band collapses onto the
  inactives' instead of being pinned high.

### Distribution

- **Every checkpoint records `pooled_ranker.library.manifest_sha256`** -- one hash naming
  the exact reference bundle the model was fitted against. The reference id cannot do that
  job: it is a promise that a published bundle never changes, and nothing enforces it. A
  bundle re-uploaded under the same id would leave every checkpoint ranking against a
  different distribution than it claims, with nothing able to detect it. The manifest pins
  each file by hash, so hashing the manifest collapses the whole bundle to one checkable
  value.

  It does not catch a changed *featurizer*. `manifest.check_environment`, the
  descriptor-drift canary, still runs only from `scripts/verify_reference_bundle.py` and
  never at fit time, so an install whose RDKit or neural checkpoints have moved will still
  fit and rank against matrices it cannot reproduce.

- **Fixed: the published URL was missing a path segment.** `PUBLIC_BASE_URL` read
  `.../lazy-qsar/reference/`, but `eosvc` puts the bundle at `.../lazy-qsar/data/reference/`
  -- `data/` is the prefix it treats as public. Any error message offering a user a download
  URL pointed at nothing. It is now built from `EOSVC_REPO` and `EOSVC_ROOT`, the same two
  constants the download itself uses, so the URL a client is pointed at is the object it
  reads. A second, also-wrong copy in `scripts/reference/config.py` that nothing read has
  been removed rather than corrected.

- **The reference library is fetched with `eosvc`**, the same tool that publishes it, so
  there is one path and one set of conventions rather than a publisher and an unrelated
  reader that can disagree. No AWS credentials are needed: `eosvc` falls back to anonymous
  access, which is all a public bucket read requires.

  `eosvc` resolves its S3 prefix from the repository it runs inside, and an installed
  package is not inside a checkout, so the client stages a minimal one and moves the files
  into the cache afterwards. The cache layout owes `eosvc` nothing.

- **`lazyqsar setup --reference [--only LIST]`**, plus a `lazyqsar reference` subcommand
  with `status`, `fetch`, `verify` and `smiles`. Not implied by `--descriptors`: a
  fast-mode model needs one 6.5 MB matrix and the whole bundle is 267 MB.

- **Matrices are fetched lazily at fit**, once the portfolio has settled which descriptors
  survive, so only what a model actually uses is downloaded.

- `LAZYQSAR_REFERENCE_OFFLINE=1` refuses to fetch instead of reaching for the network,
  which is what the test suite runs under and what an air-gapped node wants.

### Renamed

- **The out-of-fold percentile is no longer called a rank.** It is a different quantity from
  `rank`: relative to a model's own training data, not comparable across models, and not
  comparable with a position against the reference library. Both were `predict_rank`, which
  is how one gets believed to be the other. It is now `_oof_percentile` throughout the
  pipeline, where it serves its only real purpose -- weighting descriptors by how reliable
  each looks at a given percentile.

  The leaf estimators under `lazyqsar/base/` keep `predict_rank`. They are documented as
  usable independently, each owns its own ECDF, and a percentile is a reasonable thing for a
  standalone estimator to offer; the rename stops at the wrappers above them.

- **`LazyClassifier.fit` accepts `reference_X=` / `reference_h5_file=`.** Pass the
  descriptors of the molecules in `lazyqsar.reference.reference_smiles()`, computed with
  your own featurizer and in that order, and `predict_rank` becomes available on the
  descriptor-agnostic path with the same meaning it has everywhere else. A `.h5` reference
  is read in chunks, so a large one is never held whole. The reference travels through
  `save()` into the ONNX artifact, which `load()` returns.
- **`LazyClassifier.predict_rank` raises when no reference was given at fit.** It used to
  return the training-relative percentile under a name that promised a position against
  drug-like chemical space. The error names `reference_X` and `reference_smiles()`; the
  percentile itself survives as `_oof_percentile`.

- **Descriptor-level metadata keys.** `pooled_ranker` becomes `oof_percentile` and
  `decision_cutoff_rank` becomes `decision_cutoff_oof_percentile`. The first is a new key
  rather than a redefinition: `pooled_ranker` means the reference library at the task level,
  and a v3.5.x descriptor checkpoint carries out-of-fold knots under it, so reusing the name
  would let the two be read as the same thing. An older checkpoint is therefore treated as
  having no descriptor-level percentile and falls back to the graph, which is correct. The
  cutoff's meaning did not change, so its old value is still read.

- **`--predict_type rank` costs one pass, not two**, on a checkpoint without an
  applicability domain. `required_channels` no longer requests the percentile channel for a
  rank output, because a reference rank is read off the pooled probability.

### Known limitations

- **The top of the scale is anchored on known molecules.** `0.95` is the 95th percentile
  of the model's out-of-fold actives. Past the reference's p99.9 the library is too sparse
  to resolve anything -- for a selective model the whole top 1% of a screen can sit inside a
  single knot interval -- so the last stretch is pinned on molecules whose labels are known.
  An anchor that does not clear p99.9 is dropped and the scale runs straight to certainty;
  `anchor_high_used` records which branch was taken. That case is itself a finding: it means
  the model's known actives do not reach the top 0.1% of drug-like chemical space. The cost
  is that every model's top actives read 0.95 by construction, so the top of the scale does
  not distinguish model quality -- `oof_diagnostics.screening_auc` and
  `sensitivity_at_cutoff` carry that instead.
- **The bottom of the scale is flattened.** Roughly a third of a generic library can land
  below rank 0.25. That is the deliberate trade for spending the axis on the top of the
  list, but it means `rank` carries little information about *how* inactive something is,
  and nothing downstream should read low ranks quantitatively.
- **A rank says nothing on its own about model skill.** A random model still puts 1% of the
  reference above the cutoff, because the cutoff is defined that way. If the real problem is
  that top-ranked compounds are not active, this relabels it rather than fixing it -- and
  since every output is a monotone transform of one probability, none of them can separate a
  false positive from a true positive that scores the same.
- **The reference knot subsample caps the tail at p99.99.** `subsample_knots` thins 50,000
  reference probabilities to 10,000 spaced evenly *in rank*, so 7,500 of them sit below the
  third quartile -- the part of the library this scale discards. For a selective model the
  binding constraint on tail resolution is therefore the subsample, not the anchor table: its
  top 1% of compounds can fall inside one knot spacing. A tail-weighted subsample would cost
  nothing in file size and is the obvious next step.
- **`proba` remains conditioned on the training prior**, not the screening library's, so
  this release does not make probabilities real-world-calibrated.
- **The applicability-domain veto stays train-relative**, so diverse library compounds trip
  it constantly and the per-sample gating largely degenerates in real screening.

### The reference library

50,000 molecules selected from the 1,355,109-molecule `ersilia_reference_library_v0`, by
`scripts/select_reference_subset.py`. Two clusterings in two spaces: BitBIRCH on Morgan
fingerprints collapses near-duplicates, so no two reference molecules are analogues, and
MiniBatchKMeans on physchem descriptors supplies density strata. Slots per stratum are
proportional to `size ** 0.5`, which compresses crowded regions without flattening the
density that the percentile is quoted against -- proportional allocation left 52 of 10,000
strata unrepresented, and equal allocation would answer "beats 99% of chemotypes" instead.

Every selected molecule is CDDD-calculable by construction, which matters because CDDD
refuses a dataset failing more than 0.1% of its filters and the library's own rate is
1.88% -- a random 50,000-molecule slice would have made CDDD inapplicable.

The anchoring is exact by construction and checked directly: a quarter of the reference
falls above rank 0.75 and a quarter below 0.25, whatever shape the reference has.

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
- **`score` is now a monotone view of `proba` too.** It was a weighted mean of the raw
  per-descriptor scores, which -- exactly like the old `rank` -- is not a monotone
  transform of the pooled probability, so `score` and `proba` could order the same two
  molecules differently. Measured on the ChEMBL fixtures they disagreed about 581 of
  79,800 pairs, moving 151 of 233 molecules to a different position.

  The cause is not calibration reordering anything: every head's calibrator is monotone
  and never moves that head's own molecules. It is that the heads carry *different*
  curves, so calibration changes how far apart each head's opinions sit -- how loudly it
  votes -- and a weighted average of differently-stretched monotone curves is not a
  monotone function of the weighted average of the originals. No way of pooling raw
  values fixes it; four were measured and the best still left 550 flipped pairs.

  `score` is now the pooled probability read back onto the pre-calibration scale through
  a monotone map stored in the checkpoint, the same shape of fix the pooled rank
  reference is. Zero inverted pairs against `proba` by construction, and the values stay
  where they were: mean shift 0.0012, maximum 0.0318 on a scale spanning 0.06 to 0.96.
  Checkpoints fitted by earlier versions carry no map and keep their previous behaviour
  exactly; refit to get the new one.

  All six `predict_type` values now rank molecules identically.
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
- **`predict_type="score"` costs what every other output costs.** Deriving it from the
  pooled probability means the raw per-descriptor channel is never requested, and that
  channel was the last thing making any request run every preprocessor and every head
  twice. Measured end to end on real Morgan fingerprints, 8,000 molecules: `score` fell
  from 3.29s to 2.30s, matching `proba` at 2.32s, and `LazyClassifierQSAR.load(...)
  .predict_proba` -- which asks for every channel -- from 3.96s to 2.99s.
- `XGBoostArtifact.predict_score` normalises a one-column probability output to two
  columns, which `run` always did and it did not. On a legacy export -- onnxmltools
  annotating `probabilities` with `dim_value=2` while onnxruntime infers `{N,1}`, the case
  `_build_xgb_session` exists to repair -- the caller's `[:, 1]` raised, the scoring loop
  swallowed it as "channel unavailable", and `predict_type="score"` silently fell back to
  pooling the *calibrated* probabilities: a wrong number with no error anywhere. Every
  head shipped today emits two columns and is unaffected.
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
- `lazyqsar/ensemble/reference.py` gained `build_pooled_score_knots`, the score map's
  counterpart to the pooled rank reference, and `utils/ranking.py` gained
  `score_from_knots` beside `rank_from_knots`. The map is stored under `pooled_scorer` in
  the task-level `metadata.json`, next to `pooled_ranker`.
- No ONNX converter, opset or graph was changed, and nothing under `lazyqsar/base/` was
  modified beyond one import. `lazyqsar/artifacts/classifier.py` gained one method,
  `rank_from_proba`, which reads a rank off a probability the caller already has; it runs
  no graph and the values are identical to what `predict_rank` returns.
