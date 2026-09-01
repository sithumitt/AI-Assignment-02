"""
Task 04: Exploratory Data Analysis (EDA)
Generates and exports class distribution charts, clinical boxplots, and correlation heatmaps.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda_visualizations(
    df, figures_dir="reports/figures", target_col="disease_risk_level"
):
    """
    Executes exploratory analysis and exports high-resolution figure charts.
    """
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Target Class Distribution Chart
    plt.figure(figsize=(6, 4))
    sns.countplot(
        data=df,
        x=target_col,
        palette="viridis",
        order=["Low", "Medium", "High"],
    )
    plt.title("Disease Risk Level Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Risk Level")
    plt.ylabel("Patient Count")
    plt.tight_layout()
    target_fig_path = os.path.join(figures_dir, "eda_target_distribution.png")
    plt.savefig(target_fig_path, dpi=300)
    plt.close()
    print(f"[EDA Figure Saved] -> {target_fig_path}")

    # 2. Clinical Vitals vs Target Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    order = ["Low", "Medium", "High"]

    sns.boxplot(
        ax=axes[0, 0],
        data=df,
        x=target_col,
        y="blood_sugar_mg_dl",
        palette="Set2",
        order=order,
    )
    axes[0, 0].set_title("Blood Sugar vs Risk Level")

    sns.boxplot(
        ax=axes[0, 1],
        data=df,
        x=target_col,
        y="cholesterol_mg_dl",
        palette="Set2",
        order=order,
    )
    axes[0, 1].set_title("Cholesterol vs Risk Level")

    sns.boxplot(
        ax=axes[1, 0],
        data=df,
        x=target_col,
        y="bmi",
        palette="Set2",
        order=order,
    )
    axes[1, 0].set_title("BMI vs Risk Level")

    sns.boxplot(
        ax=axes[1, 1],
        data=df,
        x=target_col,
        y="systolic_bp",
        palette="Set2",
        order=order,
    )
    axes[1, 1].set_title("Systolic BP vs Risk Level")

    plt.tight_layout()
    box_fig_path = os.path.join(figures_dir, "eda_clinical_boxplots.png")
    plt.savefig(box_fig_path, dpi=300)
    plt.close()
    print(f"[EDA Figure Saved] -> {box_fig_path}")

    # 3. Numerical Clinical Correlation Heatmap
    num_cols = [
        "age",
        "previous_admissions",
        "systolic_bp",
        "diastolic_bp",
        "blood_sugar_mg_dl",
        "cholesterol_mg_dl",
        "bmi",
        "lab_tests_count",
        "treatments_count",
    ]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df[num_cols].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title(
        "Numerical Clinical Features Correlation Heatmap",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    corr_fig_path = os.path.join(figures_dir, "eda_correlation_matrix.png")
    plt.savefig(corr_fig_path, dpi=300)
    plt.close()
    print(f"[EDA Figure Saved] -> {corr_fig_path}")
