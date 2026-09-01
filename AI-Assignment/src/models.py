"""
Task 05 & Task 06: Machine Learning Development and Model Evaluation
Trains 3 classification algorithms, compares metrics, generates confusion matrices,
and serializes the best performing model.
"""

import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier


def train_and_evaluate_models(
    X_train_scaled,
    X_test_scaled,
    X_train,
    X_test,
    y_train,
    y_test,
    models_dir="models",
    figures_dir="reports/figures",
    random_state=42,
):
    """
    Trains Logistic Regression, Random Forest, and Decision Tree classifiers,
    builds comparison table, and saves the top model.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Model Definitions (algorithm, train_features, test_features)
    model_suite = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=random_state),
            X_train_scaled,
            X_test_scaled,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=random_state
            ),
            X_train,
            X_test,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=5, random_state=random_state),
            X_train,
            X_test,
        ),
    }

    results = []
    trained_clfs = {}

    # 2. Benchmark Loop
    for name, (clf, train_x, test_x) in model_suite.items():
        clf.fit(train_x, y_train)
        preds = clf.predict(test_x)
        trained_clfs[name] = clf

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")

        results.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
            }
        )

    eval_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    print("\n=== Model Benchmark Comparison ===")
    print(eval_df.to_string(index=False))

    # 3. Confusion Matrix Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    classes = ["High", "Low", "Medium"]

    for idx, (name, (clf, _, test_x)) in enumerate(model_suite.items()):
        preds = clf.predict(test_x)
        cm = confusion_matrix(y_test, preds, labels=classes)
        model_acc = eval_df.loc[eval_df["Model"] == name, "Accuracy"].values[0]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            xticklabels=classes,
            yticklabels=classes,
        )
        axes[idx].set_title(f"{name}\nAcc: {model_acc:.2%}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    plt.tight_layout()
    cm_fig_path = os.path.join(figures_dir, "model_confusion_matrices.png")
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"[Confusion Matrix Saved] -> {cm_fig_path}")

    # 4. Serialize Top Model
    best_model_name = eval_df.iloc[0]["Model"]
    best_model = trained_clfs[best_model_name]
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(
        f"[Optimal Model Selected] -> {best_model_name} (Saved to {best_model_path})"
    )

    return eval_df, best_model, best_model_name
