import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from src.features import add_features
from src.config import Config

REQUIRED_FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in REQUIRED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0  # или np.nan

    # Только нужные и в нужном порядке
    return df[REQUIRED_FEATURES]

def load_model(path):
    model = CatBoostClassifier()
    model.load_model(path)
    return model

def predict(model, df):
    df = validate_input(df)
    df = add_features(df)
    proba = model.predict_proba(df)[:, 1]
    pred = model.predict(df)
    return proba.tolist(), pred.tolist()

if __name__ == "__main__":
    config = Config()
    model = load_model(config.model_path)

    # Один пример
    sample_1 = {
        "Time": 100000,
        "Amount": 49.99,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }

    # Второй неполный (batch, с отсутствующими V)
    sample_2 = {
        "Time": 90000,
        "Amount": 100.5
    }

    input_df = pd.DataFrame([sample_1, sample_2])

    probs, preds = predict(model, input_df)

    for i, (p, c) in enumerate(zip(probs, preds)):
        print(f"→ Транзакция {i+1}: Вероятность фрода = {p:.4f}, Класс = {'FRAUD' if c==1 else 'OK'}")
