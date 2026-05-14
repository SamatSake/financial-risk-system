import json
from datetime import datetime
import os

LOG_FILE = "reports/prediction_log.jsonl"

os.makedirs("reports", exist_ok=True)


def log_prediction(input_data, prediction, decision):
    log_entry = {
        "timestamp": str(datetime.now()),
        "input": input_data,
        "prediction": prediction,
        "decision": decision
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def basic_drift_check(current_df, reference_df):
    drift_report = {}

    for col in current_df.columns:
        if col in reference_df.columns:
            drift_report[col] = {
                "mean_diff": abs(current_df[col].mean() - reference_df[col].mean())
            }

    return drift_report