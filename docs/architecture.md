# Architecture: one core, two tasks

Classification and regression share one implementation at every layer. Each layer has a
task-neutral base; the task modules add only what genuinely differs. The rule when
changing anything: put it in the base unless it is meaningless for the other task.

| Layer | Shared base | Classification adds | Regression adds |
|---|---|---|---|
| Heads (XGBoost, linear, RF, SVM) | `base/_head.py`: fold loop, export check, ranks, save. One `_<Family>Head` per family | `ClassifierHead`: calibrator, decision cutoffs, `predict_proba/score/logit`; preset races | `RegressorHead`: OOF ρ; one benchmark-selected preset per head |
| Preset race rules | `base/_race.py` (noise-adaptive margin) | used by the XGB, RF, SVC races | — |
| ONNX binding | `base/_onnx_binding.py`: fit through the graph, verify at bind, write the verified bytes | — | — |
| Head artifacts (inference) | `artifacts/_head.py` `HeadArtifact` | calibrated `run`, cutoffs | values, SVR destandardisation |
| Preprocessor | `base/preprocessing/` (one class, `task` flag) | — | — |
| Pooler | `poolers/_gate.py` `GatedPooler` / `GatedPoolerArtifact` | Bernoulli-oracle Ridge gate | NNLS stacking weights |
| Head selection | `portfolios/_base.py` `HeadSelector` | profile-scored rules | rules by dataset size |
| Assembler (one descriptor) | `assemblers/_base.py`: batches, fit, OOF reference, save | imbalance batching, prior correction, cutoffs | target standardisation, intervals |
| Feature-level API | `agnostic.py` `_AgnosticBase` | `LazyClassifier` (reference rank) | `LazyRegressor` |
| SMILES level | `qsar_base.py` `_QSARBase`: descriptor screen, one model + AD per descriptor, pruning, predict, checkpoints | `qsar.py`: reference library, `rank`, decision cutoff, diagnostics | `qsar_regressor.py`: OOF metrics, descriptor stacking, intervals |
| Combination | `ensemble/combine.py` `combine`, one weight builder | six outputs | `value`, `lower`, `upper` |
| Inference runner | `ensemble/runner.py`, `ensemble/channels.py` | — | — |

## Invariants worth knowing before a change

- **A fitted model and its checkpoint compute the same function.** Heads, preprocessor
  and applicability domain predict through their ONNX graph from the moment they are
  fitted; `save` writes exactly those bytes. Anything computed outside a graph
  (calibrators, cutoffs, stacking weights, interval half-widths) is saved as plain
  floats and applied by the same code at load. See the README section "How ONNX fidelity
  is guarded".
- **Classification outputs are pinned.** A refactor must leave every classification
  prediction bit-identical; the parity tests and the golden files are the guard.
- **Regression choices are benchmark-backed.** `dev/benchmarks/regression/` reproduces
  every number in `docs/regression.md`; re-run the relevant study before changing a
  preset, the pooler or the head selection.
- **Out-of-fold everything.** Pooler weights, descriptor weights, ranks and intervals are
  learned on out-of-fold predictions, never on in-sample ones.
