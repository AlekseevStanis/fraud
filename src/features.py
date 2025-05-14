# src/features.py
import numpy as np

def build_features(X_train, X_test, config):
    for df in [X_train, X_test]:
        df["hour_of_day"] = ((df["Time"] % 86400) // 3600).astype(int)
        df["log_amount"] = np.log1p(df["Amount"])
        df.drop(["Time", "Amount"], axis=1, inplace=True)
    return X_train, X_test
