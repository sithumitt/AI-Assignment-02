"""
Task 03: Data Preprocessing & Feature Engineering
Performs clinical feature filtering (data leakage prevention), One-Hot Encoding,
train-test splitting, and StandardScaler normalization.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_preprocessed_dataset(
    file_path_or_df="data/raw/smartcare_ai_dataset_1000.csv",
    export_csv_path="data/processed/cleaned_features.csv",
    models_dir="models",
    test_size=0.20,
    random_state=42,
):
    """
    Cleans raw dataset, encodes categoricals, scales numerical features,
    and serializes the fitted StandardScaler and model columns.
    """
    # 1. Load data
    if isinstance(file_path_or_df, str):
        df = pd.read_csv(file_path_or_df)
    else:
        df = file_path_or_df.copy()

    # 2. Define clinical feature space (excludes operational targets & billing to prevent data leakage)
    clinical_features = [
        "age",
        "gender",
        "blood_group",
        "department",
        "diagnosis",
        "previous_admissions",
        "systolic_bp",
        "diastolic_bp",
        "blood_sugar_mg_dl",
        "cholesterol_mg_dl",
        "bmi",
        "lab_tests_count",
        "treatments_count",
    ]
    target_col = "disease_risk_level"

    # Clean subset
    df_clean = df[clinical_features + [target_col]].dropna().copy()
    X_raw = df_clean[clinical_features]
    y = df_clean[target_col]

    # 3. One-Hot Encoding for categorical features
    cat_cols = ["gender", "blood_group", "department", "diagnosis"]
    num_cols = [c for c in clinical_features if c not in cat_cols]
    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    feature_names = X_encoded.columns.tolist()

    # 4. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5. Fit StandardScaler strictly on training set
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # 6. Assemble preprocessed dataset
    X_full_scaled = X_encoded.copy()
    X_full_scaled[num_cols] = scaler.transform(X_encoded[num_cols])
    df_processed = X_full_scaled.copy()
    df_processed[target_col] = y.values

    # 7. Save preprocessed CSV and transformation artifacts
    if export_csv_path:
        os.makedirs(os.path.dirname(export_csv_path), exist_ok=True)
        df_processed.to_csv(export_csv_path, index=False)
        print(f"[Preprocessed Dataset Saved] -> {export_csv_path}")

    if models_dir:
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
        joblib.dump(feature_names, os.path.join(models_dir, "model_columns.pkl"))
        print(f"[Transformation Artifacts Saved] -> {models_dir}/")

    return (
        df_processed,
        X_train_scaled,
        X_test_scaled,
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names,
    )
