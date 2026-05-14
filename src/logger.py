import json
from datetime import datetime
import os

LOG_FILE = "logs/predictions.log"


def log_prediction(input_data, prediction, decision):
    os.makedirs("logs", exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction,
        "decision": decision
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")