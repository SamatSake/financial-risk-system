import pandas as pd
from src.monitoring import basic_drift_check

current = pd.read_csv("data/processed.csv")
reference = pd.read_csv("data/processed.csv")

print(basic_drift_check(current, reference))