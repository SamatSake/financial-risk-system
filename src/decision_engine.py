def risk_decision(probability: float) -> str:

    if probability < 0.3:
        return "LOW_RISK_APPROVE"
    elif probability < 0.7:
        return "MEDIUM_RISK_MANUAL_REVIEW"
    else:
        return "HIGH_RISK_REJECT"