import shap
import joblib
import pandas as pd

MODEL_PATH = "models/model.pkl"

model = joblib.load(MODEL_PATH)

explainer = shap.TreeExplainer(model)


def explain(input_data: dict):
    df = pd.DataFrame([input_data])

    shap_values = explainer.shap_values(df)

    return {
        "feature_contributions": shap_values[1].tolist()
    }