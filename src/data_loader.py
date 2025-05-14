# src/data_loader.py
import pandas as pd

def load_data(config):
    df = pd.read_csv(config.data_path)    
    return df
