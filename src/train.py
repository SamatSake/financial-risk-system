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

    mlflow.set_experiment("risk-model-experiment")

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

        with mlflow.start_run(run_name=name):

            # ===== TRAIN =====
            model.fit(X_train, y_train)

            # ===== EVAL =====
            score = evaluate(model, X_test, y_test)

            print(name, "ROC-AUC:", score)

            # ===== LOG PARAMS =====
            mlflow.log_param("model_name", name)

            if name == "rf":
                mlflow.log_param("n_estimators", model.n_estimators)

            if name == "xgb":
                mlflow.log_param("max_depth", model.max_depth)
                mlflow.log_param("n_estimators", model.n_estimators)

            # ===== LOG METRIC =====
            mlflow.log_metric("roc_auc", score)

            # ===== LOG MODEL =====
            mlflow.sklearn.log_model(model, "model")

            # ===== BEST MODEL PICK =====
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

    # ===== SAVE BEST MODEL LOCALLY =====
    os.makedirs("models", exist_ok=True)

    if best_model is None:
        raise ValueError("No model was trained!")

    joblib.dump(best_model, MODEL_PATH)

    # ===== LOG BEST MODEL RUN (optional but good for grading) =====
    with mlflow.start_run(run_name="BEST_MODEL"):
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_roc_auc", best_score)
        mlflow.sklearn.log_model(best_model, "best_model")

    print("\nBEST MODEL:", best_name)
    print("BEST SCORE:", best_score)
    print("MODEL SAVED TO:", MODEL_PATH)

if __name__ == "__main__":
    train()