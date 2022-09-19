from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet

from wh.experiments.experiment import Experiment
from wh.pipeline.preprocess import Preprocess


class ElasticNetExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self._pipeline = Pipeline([
            ('scaler', Preprocess.scaler()),
            ('elasticnet', ElasticNet())
        ])

        self._grid_search = {
            'elasticnet__alpha': [0.5, 0.75, 1],
            'elasticnet__l1_ratio': [0, 0.2, 0.5, 0.8, 1]
        }
