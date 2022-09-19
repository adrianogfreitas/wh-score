from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from wh.experiments.experiment import Experiment
from wh.pipeline.preprocess import Preprocess


class TreeRegressorExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self._pipeline = Pipeline([
            ('scaler', Preprocess.scaler()),
            ('tree', DecisionTreeRegressor())
        ])

        self._grid_search = {
            'tree__max_depth': [4, 8, 16, 32, 64],
            'tree__min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5]
        }

