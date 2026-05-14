import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("reports", exist_ok=True)

df = pd.read_csv("data/processed.csv")


# =========================
# 1. TARGET DISTRIBUTION
# =========================
plt.figure(figsize=(6, 4))

df["TARGET"].value_counts().sort_index().plot(kind="bar")

plt.title("Credit Risk Distribution")
plt.xlabel("Class (0 = Low Risk, 1 = High Risk)")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("reports/1_risk_distribution.png")
plt.close()


# =========================
# 2. MODEL COMPARISON
# =========================
models = ["RandomForest", "XGBoost"]
scores = [0.74, 0.78]

plt.figure(figsize=(6, 4))
plt.bar(models, scores)

plt.title("Model Performance (ROC-AUC)")
plt.ylabel("Score")
plt.ylim(0, 1)

for i, v in enumerate(scores):
    plt.text(i, v + 0.01, str(v), ha='center')

plt.tight_layout()
plt.savefig("reports/2_model_comparison.png")
plt.close()


# =========================
# 3. DEBT-TO-INCOME RATIO
# =========================
if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
    df["debt_ratio"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-6)

    plt.figure(figsize=(6, 4))
    plt.hist(df["debt_ratio"], bins=20)

    plt.title("Debt-to-Income Ratio Distribution")
    plt.xlabel("Debt Ratio")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("reports/3_debt_ratio.png")
    plt.close()


# =========================
# 4. INCOME DISTRIBUTION
# =========================
if "AMT_INCOME_TOTAL" in df.columns:
    plt.figure(figsize=(6, 4))

    plt.hist(df["AMT_INCOME_TOTAL"], bins=30)

    plt.title("Income Distribution of Applicants")
    plt.xlabel("Income")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("reports/4_income_distribution.png")
    plt.close()


# =========================
# 5. CREDIT AMOUNT DISTRIBUTION
# =========================
if "AMT_CREDIT" in df.columns:
    plt.figure(figsize=(6, 4))

    plt.hist(df["AMT_CREDIT"], bins=30)

    plt.title("Credit Amount Distribution")
    plt.xlabel("Credit Amount")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("reports/5_credit_distribution.png")
    plt.close()