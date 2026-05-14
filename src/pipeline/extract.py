import pandas as pd

def extract(path="data/raw.csv"):
    return pd.read_csv(path)