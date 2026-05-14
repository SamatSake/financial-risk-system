import joblib
import pandas as pd
import os

MODEL_PATH = "models/model.pkl"

model = None


def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model


def predict_proba(input_data: dict):
    model = load_model()

    df = pd.DataFrame([input_data])

    proba = model.predict_proba(df)[0][1]

    return float(proba)