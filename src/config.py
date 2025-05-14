# src/config.py
from dataclasses import dataclass

@dataclass
class Config:
    data_path: str = "data/processed/train.csv"
    test_path: str = "data/processed/test.csv"
    model_path: str = "models/model.pkl"
    mlflow_uri: str = "http://localhost:5000"
    experiment_name: str = "fraud-detection"
    model_type: str = "xgboost"
    random_state: int = 42
    early_stopping_rounds: int = 50
    eval_metric: str = "recall"
