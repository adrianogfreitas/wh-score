from dotenv import load_dotenv
import os
import pandas as pd

from wh.config import Config
from wh.experiments.experiment import Experiment
from wh.pipeline.preprocess import Preprocess

# from wh.experiments.forest_regressor import RandomForestRegressorExperiment
from wh.experiments.svm_regressor import SVMRegressorExperiment
# from wh.experiments.xgb_regressor import XGBRegressorExperiment
# from wh.experiments.tree_regressor import TreeRegressorExperiment

def main():
    experiments = [
        # RandomForestRegressorExperiment,
        SVMRegressorExperiment,
        # TreeRegressorExperiment,
        # XGBRegressorExperiment,
    ]
    
    prep = Preprocess(data='processed')
    prep.build_scaler()

    # TODO: ajustar esse processo
    conf = Config()
    df = load_data(conf, 'processed')
    X = Preprocess.fill_na(df.drop(conf.data["target"], axis=1)).to_numpy()
    y = df[conf.data["target"]].to_numpy()

    for experiment in experiments:
        print(f'Running: {experiment}')
        experiment().run(X, y)

    exp = Experiment()
    exp.promote_best_model()

def load_data(conf, data):
    bucket = conf.data['bucket']
    folder = conf.data[f'{data}_path']
    file = conf.data[f"{data}_file"]
    data_path = f"s3://{bucket}/{folder}/{file}"

    load_dotenv()
    storage_options = {
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {
            'endpoint_url': os.getenv("MLFLOW_S3_ENDPOINT_URL")
        }
    }
    
    return pd.read_csv(data_path, storage_options=storage_options)
