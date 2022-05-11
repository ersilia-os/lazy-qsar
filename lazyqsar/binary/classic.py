from flaml import AutoML


class ClassicBinaryClassifier(object):

    def __init__(self, metric = "roc_auc", time_budget_sec=20, estimator_list = ["xgboost"]):
        self.time_budget_sec = time_budget_sec
        self.metric = metric
        self.model = AutoML(task="classification", metric=self.metric)
        self.estimator_list = estimator_list
        
    def fit(self, smiles):
        
        pass

    def predict(self, smiles):
        pass

    def predict_proba(self, smiles):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
