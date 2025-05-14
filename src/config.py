# src/config.py
from dataclasses import dataclass

@dataclass
class Config:
    data_path: str = "data/raw/creditcard.csv"
    model_path: str = "models/catboost_model.cbm"
    mlflow_uri: str = "http://localhost:5000"
    experiment_name: str = "fraud-detection"
    model_type: str = "xgboost"
    random_state: int = 42
    early_stopping_rounds: int = 50
    eval_metric: str = "recall"
