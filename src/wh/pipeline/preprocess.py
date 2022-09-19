from abc import abstractmethod
from dotenv import load_dotenv
import joblib
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
import s3fs

from wh.config import Config

class Preprocess:
    def __init__(self, data='raw'):
        """

        Parameters
            data: str = `raw` or `processed`
        """
        load_dotenv()
        self._storage_options = {
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "client_kwargs": {
                'endpoint_url': os.getenv("MLFLOW_S3_ENDPOINT_URL")
            }
        }
        
        self._conf = Config()
        self._df = self._load_data(data)
        self._col_stats = self._get_stats()

        self._na_cols = None
        self._zero_dummy_cols = None
        self._high_cardinality_cols = None
        self._high_correlated_cols = None
    
    @property
    def na_cols(self):
        """List of columns that exceeds the threshold of null values"""
        if not self._na_cols:
            total_rows = self._df.shape[0]
            max_null = self._conf.max_null_threshold * total_rows
            self._na_cols = list(self._col_stats[self._col_stats["count"] <= max_null].index)

        return self._na_cols

    @property
    def zero_dummy_cols(self):
        """List of columns that are dummy (0/1) and exceeds the threshold os 0 values"""
        if not self._zero_dummy_cols:
            dummy_cols = [col for col in self._df.columns if sorted(self._df[col].unique()) == [0,1]]
            self._zero_dummy_cols = list(self._col_stats.loc[dummy_cols].loc[self._col_stats["75%"] == 0].index)

        return self._zero_dummy_cols

    @property
    def high_cardinality_cols(self):
        if not self._high_cardinality_cols:
            self._high_cardinality_cols = list(
                self._col_stats[
                    self._df.drop(self._conf.data["target"], axis=1).nunique(axis=0) > self._conf.high_cardinality_threshold].index)

        return self._high_cardinality_cols

    @property
    def high_correlated_cols(self):
        if not self._high_correlated_cols:
            corr = self._df.corr().abs().unstack()
            redundant_pairs = self._get_redundant_pairs()
            self._high_correlated_cols = list(set(
                [pair[0] for pair in corr.drop(redundant_pairs)[corr.drop(redundant_pairs) >= self._conf.high_correlated_threshold].index]))

        return self._high_correlated_cols
    
    @abstractmethod
    def scaler():
        def _scaler(X):
            conf = Config()
            bucket = conf.artifacts['bucket']
            file = conf.artifacts[f"scaler"]
            scaler_path = f"s3://{bucket}/{file}"
            fs = s3fs.S3FileSystem(
                key = os.getenv("AWS_ACCESS_KEY_ID"),
                secret = os.getenv("AWS_SECRET_ACCESS_KEY"),
                client_kwargs = {
                    'endpoint_url': os.getenv("MLFLOW_S3_ENDPOINT_URL")
                }
            )

            with fs.open(scaler_path, 'rb') as f:
                scaler = joblib.load(f)
            
            return scaler.transform(X)
        return FunctionTransformer(_scaler)

    @abstractmethod
    def fill_na(X):
        return X.fillna(-1)

    def build_df(self):
        # bucket = self._conf.data['bucket']
        # folder = self._conf.data[f'processed_path']
        # file = self._conf.data[f"processed_file"]
        # data_path = f"s3://{bucket}/{folder}/{file}"
        data_path = self._get_data_path('processed')

        self._df.drop(set(
            self.na_cols + self.zero_dummy_cols + self.high_cardinality_cols + self.high_correlated_cols
        ), axis=1, inplace=True)

        self._save_selected_features(self._df)
        self._df.to_csv(data_path, index=False, storage_options=self._storage_options)

    def build_scaler(self):
        # bucket = self._conf.artifacts['bucket']
        # file = self._conf.artifacts[f"scaler"]
        # scaler_path = f"s3://{bucket}/{file}"
        # fs = s3fs.S3FileSystem(
        #     key = os.getenv("AWS_ACCESS_KEY_ID"),
        #     secret = os.getenv("AWS_SECRET_ACCESS_KEY"),
        #     client_kwargs = {
        #         'endpoint_url': os.getenv("MLFLOW_S3_ENDPOINT_URL")
        #     }
        # )

        X = self._df.drop(self._conf.data["target"], axis=1).to_numpy()
        min_max_scaler = MinMaxScaler().fit(X)

        self._save_artifact('scaler', min_max_scaler)

        # with fs.open(scaler_path, 'wb') as f:
        #     joblib.dump(min_max_scaler, f)
    
    def _save_selected_features(self, df):
        features = list(df.drop(self._conf.data["target"], axis=1).columns)
        self._save_artifact('features', features)
    
    def _get_redundant_pairs(self):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = self._df.columns
        for i in range(0, self._df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return list(pairs_to_drop)

    def _load_data(self, stage):
        # bucket = self._conf.data['bucket']
        # folder = self._conf.data[f'{data}_path']
        # file = self._conf.data[f"{data}_file"]
        # data_path = f"s3://{bucket}/{folder}/{file}"
        data_path = self._get_data_path(stage)
        
        return pd.read_csv(data_path, storage_options=self._storage_options)

    def _get_stats(self):
        stats = self._df.describe().T
        stats.drop(self._conf.data['target'], axis=0, inplace=True)
        return stats

    def _get_data_path(self, stage):
        bucket = self._conf.data['bucket']
        folder = self._conf.data[f'{stage}_path']
        file = self._conf.data[f"{stage}_file"]
        return f"s3://{bucket}/{folder}/{file}"
    
    def _save_artifact(self, artifact_type, artifact):
        bucket = self._conf.artifacts['bucket']
        file = self._conf.artifacts[artifact_type]
        s3_path = f"s3://{bucket}/{file}"
        fs = s3fs.S3FileSystem(
            key = os.getenv("AWS_ACCESS_KEY_ID"),
            secret = os.getenv("AWS_SECRET_ACCESS_KEY"),
            client_kwargs = {
                'endpoint_url': os.getenv("MLFLOW_S3_ENDPOINT_URL")
            }
        )

        with fs.open(s3_path, 'wb') as f:
            joblib.dump(artifact, f)
