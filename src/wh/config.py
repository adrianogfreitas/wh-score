from dataclasses import dataclass, field

@dataclass
class Config:
    data: dict = field(
        default_factory=lambda: {
            "bucket": "data",
            "raw_path": "raw",
            "raw_file": "dataset_00_with_header.zip",
            "processed_path": "processed",
            "processed_file": "processed_dataset.csv",
            "target": "y"
        }
    )
    artifacts: dict = field(
        default_factory=lambda :{
            "bucket": "artifacts",
            "scaler": "scaler.joblib",
            "features": "features.joblib"
        }
    )
    max_null_threshold = 0.4
    high_cardinality_threshold = 100
    high_correlated_threshold = 0.75
    project_name = 'WHScore'
    model_name = "wh-score"
