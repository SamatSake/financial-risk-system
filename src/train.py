import pandas as pd
import joblib
import os

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import FEATURES, TARGET


MODEL_PATH = "models/model.pkl"


def get_models():
    return {
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgb": XGBClassifier(eval_metric="logloss", random_state=42)
    }


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, proba)


def train():

    df = pd.read_csv("data/processed.csv")

    print("DATA SHAPE:", df.shape)

    if df.empty:
        raise ValueError("Dataset is empty!")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_models()

    best_model = None
    best_score = -1
    best_name = None

    for name, model in models.items():

        model.fit(X_train, y_train)

        score = evaluate(model, X_test, y_test)

        print(name, "ROC-AUC:", score)

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    # 🔥 GUARANTEED SAVE
    os.makedirs("models", exist_ok=True)

    if best_model is None:
        raise ValueError("No model was trained!")

    joblib.dump(best_model, MODEL_PATH)

    print("\nBEST MODEL:", best_name)
    print("BEST SCORE:", best_score)
    print("MODEL SAVED TO:", MODEL_PATH)


if __name__ == "__main__":
    train()