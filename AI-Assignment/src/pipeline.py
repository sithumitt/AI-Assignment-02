"""
End-to-End Execution Pipeline
Orchestrates Tasks 02 through 07 in a single executable script.
"""

from src.dataset import inspect_dataset_quality, load_raw_dataset
from src.eda import run_eda_visualizations
from src.explainability import generate_shap_analysis
from src.models import train_and_evaluate_models
from src.preprocessing import get_preprocessed_dataset


def run_full_pipeline():
    print("==================================================")
    print(" SmartCare AI: Disease Risk ML Pipeline")
    print("==================================================")

    # Task 02: Ingestion & Understanding
    raw_df = load_raw_dataset("data/raw/smartcare_ai_dataset_1000.csv")
    inspect_dataset_quality(raw_df)

    # Task 03: Preprocessing & Feature Engineering
    (
        df_processed,
        X_train_scaled,
        X_test_scaled,
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names,
    ) = get_preprocessed_dataset(
        raw_df,
        export_csv_path="data/processed/cleaned_features.csv",
        models_dir="models",
    )

    # Task 04: EDA Visualizations
    run_eda_visualizations(raw_df, figures_dir="reports/figures")

    # Task 05 & 06: Model Development & Evaluation
    eval_df, best_model, best_model_name = train_and_evaluate_models(
        X_train_scaled,
        X_test_scaled,
        X_train,
        X_test,
        y_train,
        y_test,
        models_dir="models",
        figures_dir="reports/figures",
    )

    # Task 07: Explainable AI
    generate_shap_analysis(
        best_model,
        best_model_name,
        X_train_scaled,
        X_test_scaled,
        X_test,
        feature_names,
        figures_dir="reports/figures",
    )

    print("\n[SUCCESS] Entire AI pipeline executed and artifacts exported.")


if __name__ == "__main__":
    run_full_pipeline()
