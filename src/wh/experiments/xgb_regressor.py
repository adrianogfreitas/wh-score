from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from wh.experiments.experiment import Experiment
from wh.pipeline.preprocess import Preprocess


class XGBRegressorExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self._pipeline = Pipeline([
            ('scaler', Preprocess.scaler()),
            ('xgb', XGBRegressor())
        ])

        self._grid_search = {}

