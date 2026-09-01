"""
Task 07: Explainable AI (SHAP Analysis)
Computes Shapley additive explanations to provide global feature importance interpretability.
"""

import os
import matplotlib.pyplot as plt
import shap


def generate_shap_analysis(
    best_model,
    best_model_name,
    X_train_scaled,
    X_test_scaled,
    X_test,
    feature_names,
    figures_dir="reports/figures",
):
    """
    Constructs SHAP explainer, computes SHAP values, and saves summary plot.
    """
    os.makedirs(figures_dir, exist_ok=True)

    # Choose explainer based on model architecture
    if "Forest" in best_model_name or "Tree" in best_model_name:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        test_sample = X_test
    else:
        # Linear/Logistic Regression Explainer
        explainer = shap.LinearExplainer(best_model, X_train_scaled)
        shap_values = explainer.shap_values(X_test_scaled)
        test_sample = X_test_scaled

    # Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        test_sample,
        feature_names=feature_names,
        show=False,
    )
    plt.title("SHAP Global Feature Importance", fontsize=12, fontweight="bold")
    plt.tight_layout()

    shap_fig_path = os.path.join(figures_dir, "shap_summary_plot.png")
    plt.savefig(shap_fig_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[SHAP Summary Plot Saved] -> {shap_fig_path}")

    return explainer, shap_values
