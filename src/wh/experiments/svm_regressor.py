from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from wh.experiments.experiment import Experiment
from wh.pipeline.preprocess import Preprocess


class SVMRegressorExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self._pipeline = Pipeline([
            ('scaler', Preprocess.scaler()),
            ('svr', SVR())
        ])

        self._grid_search = {
            'svr__C': [0.3, 0.5, 0.8, 1.0],
        }

