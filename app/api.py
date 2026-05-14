from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_proba
from src.decision_engine import risk_decision
from src.explain import explain
from src.monitoring.monitoring import log_prediction

app = FastAPI(title="Financial Risk Scoring System")


class ClientData(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    avg_credit_sum: float
    avg_credit_debt: float
    avg_days_overdue: float
    previous_credit_count: float

@app.post("/explain")
def explain_prediction(data: ClientData):
    try:
        input_dict = data.dict()

        explanation = explain(input_dict)

        return {
            "status": "success",
            "explanation": explanation
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
@app.post("/predict")
def predict(data: ClientData):

    input_dict = data.dict()

    probability = float(predict_proba(input_dict))
    decision = risk_decision(probability)

    try:
        log_prediction(input_dict, probability, decision)
    except Exception as e:
        print("Logging failed:", e)

    return {
        "risk_score": probability,
        "decision": decision
    }