import pandas as pd
from src.config import Config
from predict import load_model, predict

def test_predict_output_shape():
    config = Config()
    model = load_model(config.model_path)

    sample = {
        "Time": 12345,
        "Amount": 99.99,
        **{f"V{i}": 0 for i in range(1, 29)}
    }

    df = pd.DataFrame([sample])
    proba, pred = predict(model, df)

    assert len(proba) == 1
    assert len(pred) == 1
    assert 0 <= proba[0] <= 1
    assert pred[0] in [0, 1]
