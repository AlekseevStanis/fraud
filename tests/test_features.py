import pandas as pd
from src.features import build_features
from src.config import Config
def test_add_features():
    df = pd.DataFrame([{
        "Time": 3600,
        "Amount": 10.0,
        **{f"V{i}": 0 for i in range(1, 29)}
    }])
    config = Config() 
    transformed = build_features(df, config, predict=1)
    assert "hour_of_day" in transformed.columns
    assert "log_amount" in transformed.columns
    assert "Time" not in transformed.columns
    assert "Amount" not in transformed.columns
