from xgboost import XGBClassifier

def get_model():
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss"
    )