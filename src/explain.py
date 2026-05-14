def explain(input_dict, prediction=None):

    credit = input_dict["AMT_CREDIT"]
    income = input_dict["AMT_INCOME_TOTAL"]
    history = input_dict["previous_credit_count"]

    reasons = []

    if credit / max(income, 1) > 0.5:
        reasons.append("High credit-to-income ratio increases risk")

    if history < 2:
        reasons.append("Very limited credit history")

    if income < 30000:
        reasons.append("Low income level increases default probability")

    return {
        "decision_explanation": (
            "HIGH RISK" if len(reasons) >= 2 else "MODERATE RISK"
        ),
        "main_reasons": reasons,
        "feature_impacts": {
            "AMT_CREDIT": credit,
            "AMT_INCOME_TOTAL": income,
            "previous_credit_count": history
        }
    }