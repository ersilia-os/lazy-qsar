from tunetables_light.scripts.transformer_prediction_interface import TuneTablesClassifierLight as TuneTablesClassifierLightBase
from tunetables_light.scripts.transformer_prediction_interface import TuneTablesZeroShotClassifier as TuneTablesZeroShotClassifierBase

class TuneTablesClassifierLight(TuneTablesClassifierLightBase):
    def __init__(
            self, 
            device="cpu",
            epoch=4,
            batch_size=4,
            lr=0.1,
            dropout=0.2,
            tuned_prompt_size=10,
            early_stopping=4,
            boosting=False,
            bagging=False, 
            ensemble_size=5,
            average_ensemble=False
            ):
        super().__init__(
            lr=lr, 
            epoch=epoch, 
            device=device, 
            tuned_prompt_size=tuned_prompt_size, 
            batch_size=batch_size,
            dropout=dropout,
            early_stopping=early_stopping,
            boosting=boosting,
            bagging=bagging,
            ensemble_size=ensemble_size,
            average_ensemble=average_ensemble
        )
    def predict(self, X):
        return super().predict_proba(X)[:,1]

class TuneTablesZeroShotClassifier(TuneTablesZeroShotClassifierBase):
    def __init__(self, subsample_features=True):
        super().__init__(subsample_features=subsample_features)