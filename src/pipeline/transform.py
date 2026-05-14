import pandas as pd


def transform_data():

    # MAIN APPLICATION DATA
    app_df = pd.read_csv("data/application_train.csv")

    # BUREAU DATA
    bureau_df = pd.read_csv("data/bureau.csv")

    # -----------------------------
    # FEATURE ENGINEERING FROM BUREAU
    # -----------------------------

    bureau_agg = bureau_df.groupby("SK_ID_CURR").agg({
        "AMT_CREDIT_SUM": "mean",
        "AMT_CREDIT_SUM_DEBT": "mean",
        "CREDIT_DAY_OVERDUE": "mean",
        "SK_ID_BUREAU": "count"
    }).reset_index()

    bureau_agg.columns = [
        "SK_ID_CURR",
        "avg_credit_sum",
        "avg_credit_debt",
        "avg_days_overdue",
        "previous_credit_count"
    ]

    # -----------------------------
    # MERGE TABLES
    # -----------------------------

    df = app_df.merge(
        bureau_agg,
        on="SK_ID_CURR",
        how="left"
    )

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------

    df.fillna(0, inplace=True)

    # -----------------------------
    # SELECT FEATURES
    # -----------------------------

    selected_columns = [
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "avg_credit_sum",
        "avg_credit_debt",
        "avg_days_overdue",
        "previous_credit_count"
    ]

    df = df[selected_columns]

    # -----------------------------
    # SAVE PROCESSED DATA
    # -----------------------------

    df.to_csv("data/processed.csv", index=False)

    print("Processed dataset saved.")