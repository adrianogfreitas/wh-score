from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from wh.experiments.experiment import Experiment
from wh.pipeline.preprocess import Preprocess


class RandomForestRegressorExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self._pipeline = Pipeline([
            ('scaler', Preprocess.scaler()),
            ('tree', RandomForestRegressor(criterion='squared_error'))
        ])

        self._grid_search = {
            'tree__n_estimators': [10, 50, 100, 200],
            'tree__max_depth': [4, 8, 16, 32, 64],
            'tree__min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5]
        }

