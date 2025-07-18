"""
import multiprocessing
import optuna
import numpy as np
import joblib

from sklearn.linear_model import RidgeCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


NUM_CPU = max(1, int(multiprocessing.cpu_count()/2))


class BaseLinearRegressor(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        pca: bool = False,
        num_splits: int = 3,
        test_size: float = 0.25,
        random_state: int = 42,
        num_trials: int = 50,
        timeout: int = 600,
        max_positive_proportion: float = 0.5
    ):
        self.pca = pca
        self.random_state = random_state
        self.num_splits = num_splits
        self.test_size = test_size
        self.num_trials = num_trials
        self.timeout = timeout
        self.max_positive_proportion = max_positive_proportion
        self.mean_score_ = None

    def _objective(self, trial, X, y):

        n_components = trial.suggest_float("n_components", 0.80, 0.99, step=0.01)

        num_splits = max(self.num_splits, int(1 / self.test_size))

        cv = StratifiedKFolder(n_splits=num_splits, max_positive_proportion=self.max_positive_proportion, random_state=self.random_state)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("reg", RidgeCV(refit=False, n_jobs=NUM_CPU,
                            cv=cv, random_state=self.random_state,
                            scoring="r2"))
        ])

        scores = []
        for _ in range(3):
            scores += [cross_val_score(pipe, X, y, cv=cv, scoring="r2", n_jobs=NUM_CPU).mean()]

        return np.mean(scores)
    
    def _suggest_best_params(self, X, y, test_size):

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

        print("Fitting...")
        best_score = -np.inf
        trials_without_improvement = 0
        improvement_threshold = 0.01
        patience = max(5, self.num_trials // 5)
        early_stopping = False
        baseline_score_for_patience = best_score

        def objective_with_custom_early_stop(trial):
            nonlocal best_score, trials_without_improvement, early_stopping, baseline_score_for_patience
            if early_stopping:
                print("Skipping trial due to early stopping criteria.")
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
                print(f"Early stopping: No significant improvement in the last {patience} trials.")
                raise optuna.exceptions.TrialPruned()
            return score
        
        study.optimize(
            objective_with_custom_early_stop,
            n_trials=self.num_trials,
            timeout=self.timeout
        )

        results = {
            "best_params": study.best_params,
            "best_value": study.best_value,
        }
        
        return results

    def _fit_pca(self, X, y):
        results = self._suggest_best_params(X, y, self.test_size)
        n_components = results['best_params']['n_components']
        hyperparams = results['best_params']
        score = results['best_value']
        hyperparams['n_jobs'] = NUM_CPU
        hyperparams['random_state'] = self.random_state
        hyperparams['class_weight'] = 'balanced'
        print(f"Best hyperparameters: {hyperparams}, Inner hyperparameter AUROC: {score}")

        print("Fitting on data with shape:", X.shape)
        scores = []
        num_splits = self.num_splits
        test_size = self.test_size
        if num_splits is None:
            num_splits = 3
        if test_size is None:
            test_size = 0.25
        num_splits = max(num_splits, int(1 / test_size))
        cv = StratifiedKFolder(n_splits=num_splits, shuffle=True, random_state=self.random_state, max_positive_proportion=self.max_positive_proportion)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("reg", RidgeCV(refit=True, n_jobs=NUM_CPU,
                            cv=cv, random_state=self.random_state,
                            scoring="r2"))
        ])
    
        pipe.fit(X, y)
        model_cv = pipe.named_steps["clf"]
        if hasattr(model_cv, "scores_"):
            scores_dict = model_cv.scores_
            if 1 in scores_dict:
                scores = scores_dict[1].mean(axis=1)
            else:
                scores = list(scores_dict.values())[0].mean(axis=1)
        else:
            scores = []
        print(f"Logistic regression fit done.")
        self.mean_score_ = np.mean(scores)
        self.std_score_ = np.std(scores)
        print(f"Average AUROC: {self.mean_score_}")

        best_C = model_cv.C_[0]
        solver = model_cv.solver
        penalty = model_cv.penalty
        fit_intercept = model_cv.fit_intercept
        class_weight = model_cv.class_weight
        intercept_scaling = getattr(model_cv, 'intercept_scaling', 1)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("reg", LogisticRegression(
                C=best_C,
                solver=solver,
                penalty=penalty,
                fit_intercept=fit_intercept,
                class_weight=class_weight,
                intercept_scaling=intercept_scaling,
                max_iter=1000
            ))
        ])
        pipe.fit(X, y)
        self.model_ = pipe
        return self
    
    def _fit_no_pca(self, X, y):
        scores = []
        num_splits = self.num_splits
        test_size = self.test_size
        if num_splits is None:
            num_splits = 3
        if test_size is None:
            test_size = 0.25
        num_splits = max(num_splits, int(1 / test_size))
        cv = cv = StratifiedKFolder(n_splits=num_splits, shuffle=True, random_state=self.random_state, max_positive_proportion=self.max_positive_proportion)
        model_cv = LogisticRegressionCV(
            class_weight="balanced",
            refit=True,
            n_jobs=NUM_CPU,
            cv=cv,
            random_state=self.random_state,
            scoring="roc_auc"
        )
        model_cv.fit(X, y)
        if hasattr(model_cv, "scores_"):
            scores_dict = model_cv.scores_
            if 1 in scores_dict:
                scores = scores_dict[1].mean(axis=1)
            else:
                scores = list(scores_dict.values())[0].mean(axis=1)
        else:
            scores = []
        print(f"Logistic regression fit done.")
        self.mean_score_ = np.mean(scores)
        self.std_score_ = np.std(scores)
        print(f"Average AUROC: {self.mean_score_}")

        best_C = model_cv.C_[0]
        solver = model_cv.solver
        penalty = model_cv.penalty
        fit_intercept = model_cv.fit_intercept
        class_weight = model_cv.class_weight
        intercept_scaling = getattr(model_cv, 'intercept_scaling', 1)
        model = LogisticRegression(
                    C=best_C,
                    solver=solver,
                    penalty=penalty,
                    fit_intercept=fit_intercept,
                    class_weight=class_weight,
                    intercept_scaling=intercept_scaling,
                    max_iter=1000
                )
        model.fit(X, y)
        self.model_ = model
        return self

    def fit(self, X, y):
        if self.pca:
            return self._fit_pca(X, y)
        else:
            return self._fit_no_pca(X, y)

    def predict_proba(self, X):
        if not hasattr(self, "model_") or self.model_ is None:
            raise ValueError("Model not fitted. Call `fit` first.")
        y_hat = self.model_.predict_proba(X)[:,1]
        return np.vstack((1 - y_hat, y_hat)).T
    
    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
    
    def score(self, X, y):
        return roc_auc_score(y, self.predict_proba(X)[:, 1])

    def save_model(self, model_dir: str):
        if not hasattr(self, "model_") or self.model_ is None:
            raise ValueError("Model not fitted. Call `fit` first.")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(self.model_, model_path)

        metadata = {
            "random_state": self.random_state,
            "num_splits": self.num_splits,
            "test_size": self.test_size,
            "num_trials": self.num_trials,
            "mean_score_": getattr(self, "mean_score_", None),
            "std_score_": getattr(self, "std_score_", None)
        }
        meta_path = os.path.join(model_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load_model(cls, model_dir: str):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

        model_path = os.path.join(model_dir, "model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        model = joblib.load(model_path)
        meta_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file {meta_path} not found.")
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        obj = cls(
            random_state=metadata["random_state"],
            num_splits=metadata["num_splits"],
            test_size=metadata["test_size"],
            num_trials=metadata["num_trials"]
        )
        obj.model_ = model
        obj.mean_score_ = metadata.get("mean_score_", None)
        obj.std_score_ = metadata.get("std_score_", None)

        return obj
"""