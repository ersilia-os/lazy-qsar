import numpy as np
import multiprocessing
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import optuna
from .logging import logger
from .samplers import StratifiedKFolder


NUM_CPU = max(1, int(multiprocessing.cpu_count() / 2))


class PCADimensionsOptimizerForBinaryClassification(object):
    def __init__(
        self,
        num_splits: int = 3,
        test_size: float = 0.25,
        random_state: int = 42,
        num_trials: int = 5,
        timeout: int = 600,
        max_positive_proportion: float = 0.5,
    ):
        self.test_size = test_size
        self.num_splits = num_splits
        self.max_positive_proportion = max_positive_proportion
        self.random_state = random_state
        self.num_trials = num_trials
        self.timeout = timeout

    def _objective(self, trial, X, y):
        n_components = trial.suggest_float("n_components", 0.80, 0.99, step=0.01)

        num_splits = max(self.num_splits, int(1 / self.test_size))

        cv = StratifiedKFolder(
            test_size=self.test_size,
            n_splits=num_splits,
            max_positive_proportion=self.max_positive_proportion,
            random_state=self.random_state,
        )

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                (
                    "clf",
                    LogisticRegressionCV(
                        class_weight="balanced",
                        refit=False,
                        n_jobs=NUM_CPU,
                        cv=cv,
                        random_state=self.random_state,
                        scoring="roc_auc",
                        max_iter=1000
                    ),
                ),
            ]
        )

        scores = []
        for _ in range(3):
            scores += [
                cross_val_score(
                    pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=NUM_CPU
                ).mean()
            ]

        return np.mean(scores)

    def _suggest_best_params(self, X, y):
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction="maximize",
            study_name=None,
            sampler=sampler,
        )

        n_components = 0.99
        hyperparams = {
            "n_components": n_components,
        }

        study.enqueue_trial(hyperparams)

        logger.debug(
            f"Starting hyperparameter search "
            f"sampler=TPESampler(seed={self.random_state}), direction=maximize, "
            f"trials={self.num_trials}, init:n_components={n_components}, "
            f"early_stop(threshold={improvement_threshold:.3f}, patience={patience}), "
            f"data=X{getattr(X, 'shape', None)}, y_len={len(y)}"
        )
        best_score = -np.inf
        trials_without_improvement = 0
        improvement_threshold = 0.01
        patience = max(5, self.num_trials // 5)
        early_stopping = False
        baseline_score_for_patience = best_score

        def objective_with_custom_early_stop(trial):
            nonlocal \
                best_score, \
                trials_without_improvement, \
                early_stopping, \
                baseline_score_for_patience
            if early_stopping:
                logger.debug("Skipping trial due to early stopping criteria.")
                raise optuna.exceptions.TrialPruned()
            score = self._objective(trial, X, y)
            if score > best_score:
                best_score = score
            if score > baseline_score_for_patience + improvement_threshold:
                trials_without_improvement = 0
                baseline_score_for_patience = score
            else:
                trials_without_improvement += 1
            if trials_without_improvement >= patience:
                early_stopping = True
                logger.debug(
                    f"Early stopping: No significant improvement in the last {patience} trials."
                )
                raise optuna.exceptions.TrialPruned()
            return score

        study.optimize(
            objective_with_custom_early_stop,
            n_trials=self.num_trials,
            timeout=self.timeout,
        )

        results = {
            "best_params": study.best_params,
            "best_value": study.best_value,
        }
        return results

    def suggest(self, X, y):
        if self.num_trials == 0:
            logger.info(
                f"[bold cyan]No hyperparameter trials requested[/] (num_trials={self.num_trials}); "
                f"returning defaults: best_params={{'n_components': 0.90}}, best_value=None, "
                f"data=X{getattr(X,'shape',None)}, y_len={len(y)}"
            )
            return {
                "best_params": {"n_components": 0.90},
                "best_value": None,
            }
        return self._suggest_best_params(X, y)
