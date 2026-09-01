"""
Task 02: Dataset Understanding & Auditing
Handles dataset ingestion, data dictionary verification, and quality audit.
"""

import pandas as pd


def load_raw_dataset(csv_path="data/raw/smartcare_ai_dataset_1000.csv"):
    """
    Loads raw CSV dataset into a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    print(f"[Dataset Ingested] Total Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    return df


def inspect_dataset_quality(df):
    """
    Performs data quality checks: missing values, data types, and target class balance.
    """
    print("\n--- Target Variable Distribution ---")
    if "disease_risk_level" in df.columns:
        print(df["disease_risk_level"].value_counts(dropna=False))

    print("\n--- Missing Values Summary ---")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        print(missing_cols)
    else:
        print("No missing values detected.")

    return {
        "shape": df.shape,
        "missing_counts": missing.to_dict(),
        "columns": df.columns.tolist(),
    }
