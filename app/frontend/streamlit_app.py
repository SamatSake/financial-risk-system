import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.title("Financial Risk Scoring System")

st.write("Enter client data")

age = st.number_input("Age", min_value=18, max_value=100, value=25)
income = st.number_input("Income", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
credit_history = st.number_input("Previous Credits", min_value=0, value=5)

if st.button("Predict Risk"):

    payload = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": loan_amount,
        "AMT_ANNUITY": loan_amount * 0.1,
        "DAYS_BIRTH": -age * 365,
        "DAYS_EMPLOYED": -5 * 365,
        "avg_credit_sum": loan_amount,
        "avg_credit_debt": loan_amount * 0.2,
        "avg_days_overdue": 0,
        "previous_credit_count": credit_history
    }

    try:
        # 1. PREDICT
        res = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if res.status_code == 200:
            result = res.json()
            st.success("Prediction received")
            st.json(result)

        # 2. EXPLAIN
        exp = requests.post("http://127.0.0.1:8000/explain", json=payload)

        if exp.status_code == 200:

            data = exp.json()["explanation"]

            st.subheader("Decision Explanation")
            st.write(data["decision_explanation"])

            st.subheader("Main Reasons")

            for r in data["main_reasons"]:
                st.write("•", r)

            # VISUALIZATION (important for 2.6)
            st.subheader("Feature Values (Explainability)")

            df = pd.DataFrame({
                "feature": list(data["feature_impacts"].keys()),
                "value": list(data["feature_impacts"].values())
            })

            fig, ax = plt.subplots()
            ax.barh(df["feature"], df["value"])
            ax.set_title("Risk Feature Overview")

            st.pyplot(fig)

        else:
            st.error(f"Explain API error: {exp.status_code}")
            st.text(exp.text)

    except Exception as e:
        st.error(f"Connection error: {e}")