from abc import abstractmethod
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.model_selection import GridSearchCV

from wh.config import Config
from wh.metrics import Metrics

class Experiment():
    """Base class for experiments
    
    For each experiment you should inherit from this class and define both `self._pipeline`
    and `self._grid_search`
    """
    def __init__(self):
        self._conf = Config()
        self._mlflow_client = MlflowClient()
        self._metrics = {
            'RMSE': Metrics.rmse_error(), 
            'Accuracy': Metrics.custom_accuracy_metric()
        }
        
        # Should be defined in child classes
        self._pipeline = None
        self._grid_search = None
    
    @property
    def estimator(self):
        cv = GridSearchCV(self._pipeline, self._grid_search, scoring=self._metrics, n_jobs=2, refit='Accuracy')

        return cv

    def run(self, X, y):
        mlflow.autolog()
        mlflow.set_experiment(self._conf.project_name)
        with mlflow.start_run() as run:
            self.estimator.fit(X, y)
            print("Logged run_id: {}".format(run.info.run_id))
    
    def promote_best_model(self):
        runs = self._get_runs()
        best_run_id = runs[runs['run_type'] == 'parent'].sort_values(by='training_rmse', ascending=True).reset_index().iloc[0]['id']
        
        mlflow.register_model(
            f"runs:/{best_run_id}/model",
            self._conf.model_name
        )

        self._mlflow_client=MlflowClient()
        latest_version = self._mlflow_client.get_latest_versions('wh-score', stages=['None'])[-1].version
        self._mlflow_client.transition_model_version_stage(name='wh-score', version=latest_version, stage='Production')

    def _get_runs(self, only_root: bool = False, run_name: str = None) -> pd.DataFrame:
        rows = []
        mlflow_exp = mlflow.get_experiment_by_name(self._conf.project_name)
        for run_info in self._mlflow_client.list_run_infos(mlflow_exp.experiment_id):
            run = {}

            run['id'] = run_info.run_id
            run_data = self._mlflow_client.get_run(run_info.run_id).data

            run['parent'] = run_data.tags.get('mlflow.parentRunId') or run['id']  # 'ROOT'

            if only_root and run['parent'] != run['id']:  # 'ROOT':
                continue

            if run['parent'] == run['id']:  # 'ROOT':
                run['name'] = run_data.tags.get('mlflow.runName')
            else:
                run['name'] = self._mlflow_client.get_run(run['parent']).data.tags.get('mlflow.runName')

            if run_name and run_name != run['name']:
                continue

            run['estimator_name'] = run_data.tags.get('estimator_name')
            run.update(run_data.params)
            run.update(run_data.metrics)
            rows.append(run)

            df = pd.DataFrame(rows)
            df['run_type'] = df.apply(lambda row: 'parent' if row['parent'] == row['id'] else 'run', axis=1)

        return df.sort_values(by=['name', 'parent', 'run_type', 'id']).set_index(['parent', 'id'])
