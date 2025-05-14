# src/features.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def build_features(df, config, predict=0):
    
    df["hour_of_day"] = ((df["Time"] % 86400) // 3600).astype(int)
    df["log_amount"] = np.log1p(df["Amount"])
    df.drop(["Time", "Amount"], axis=1, inplace=True)
    
    if predict==0:
        X = df.drop("Class", axis=1)
        y = df["Class"]
        # Стратифицированный сплит с учётом редкости фрода
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=config.random_state
        )
        return X_train, X_test, y_train, y_test
    else:
        return df