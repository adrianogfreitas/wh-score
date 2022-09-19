from dotenv import load_dotenv
import joblib
import os
import pandas as pd
import requests
import sys
import s3fs

from wh.config import Config

def predict(filename: str):
    features  = load_artifact('features')

    df = pd.read_csv(filename)
    X = df[features].head().fillna(-1).to_json(orient='split')

    url = 'http://localhost:5001/invocations'
    headers = {'Content-Type': 'application/json'}

    r = requests.post(url, headers=headers, data=X)

    preds = pd.DataFrame({
        'predictions': [int(score) for score in r.json()]
    })
    preds.to_csv('predictions.csv')

    return 0

def load_artifact(artifact_name):
    load_dotenv()
    conf = Config()
    bucket = conf.artifacts['bucket']
    file = conf.artifacts[artifact_name]
    s3_path = f"s3://{bucket}/{file}"
    fs = s3fs.S3FileSystem(
        key = os.getenv("AWS_ACCESS_KEY_ID"),
        secret = os.getenv("AWS_SECRET_ACCESS_KEY"),
        client_kwargs = {
            'endpoint_url': os.getenv("MLFLOW_S3_ENDPOINT_URL")
        }
    )

    with fs.open(s3_path, 'rb') as f:
        features = joblib.load(f)

    return features

def main():
    if len(sys.argv) < 2:
        raise ValueError("Please inform the file path to make the predictions")
        return

    filename = sys.argv[1]
    return predict(filename)
