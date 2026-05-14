from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_proba
from src.decision_engine import risk_decision
from src.explain import explain

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

    input_dict = data.dict()

    explanation = explain(input_dict)

    return explanation

@app.post("/predict")
def predict(data: ClientData):

    input_dict = data.dict()

    probability = predict_proba(input_dict)
    decision = risk_decision(probability)

    return {
        "risk_score": probability,
        "decision": decision
    }