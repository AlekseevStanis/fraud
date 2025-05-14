# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(config):
    df = pd.read_csv(config.data_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    # Стратифицированный сплит с учётом редкости фрода
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.random_state
    )
    return X_train, X_test, y_train, y_test
