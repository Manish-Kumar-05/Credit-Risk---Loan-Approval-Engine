from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "loan_data.csv"
MODEL_PATH = BASE_DIR / "models" / "final_knn_pipeline.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "decision_threshold.joblib"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# TARGET
# ============================================================

TARGET = "loan_status"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

X = X.copy()

X["loan_to_income"] = (
    X["loan_amnt"] /
    X["person_income"].replace(0, pd.NA)
)

X["loan_to_income"] = X["loan_to_income"].fillna(0)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print(f"Loaded model: {MODEL_PATH.name}")


# ============================================================
# LOAD PRODUCTION THRESHOLD
# ============================================================

threshold = joblib.load(THRESHOLD_PATH)

print(f"Production threshold: {threshold}")


# ============================================================
# MODEL PREDICTIONS
# ============================================================

y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# DEFAULT MODEL PREDICTION
# ============================================================

y_pred_default = model.predict(X_test)


# ============================================================
# PRODUCTION THRESHOLD PREDICTION
# ============================================================

y_pred_threshold = (
    y_probability >= threshold
).astype(int)


# ============================================================
# DEFAULT METRICS
# ============================================================

default_accuracy = accuracy_score(
    y_test,
    y_pred_default,
)

default_precision = precision_score(
    y_test,
    y_pred_default,
    zero_division=0,
)

default_recall = recall_score(
    y_test,
    y_pred_default,
    zero_division=0,
)

default_f1 = f1_score(
    y_test,
    y_pred_default,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    y_probability,
)


# ============================================================
# PRODUCTION THRESHOLD METRICS
# ============================================================

threshold_accuracy = accuracy_score(
    y_test,
    y_pred_threshold,
)

threshold_precision = precision_score(
    y_test,
    y_pred_threshold,
    zero_division=0,
)

threshold_recall = recall_score(
    y_test,
    y_pred_threshold,
    zero_division=0,
)

threshold_f1 = f1_score(
    y_test,
    y_pred_threshold,
    zero_division=0,
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

cm_default = confusion_matrix(
    y_test,
    y_pred_default,
)

cm_threshold = confusion_matrix(
    y_test,
    y_pred_threshold,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("CREDIT RISK ENGINE - EVALUATION")
print("=" * 65)

print("\nModel:")
print("Tuned K-Nearest Neighbors")

print(f"\nTest samples: {len(y_test)}")

print(f"ROC-AUC: {roc_auc:.4f}")


print("\nDefault threshold metrics")
print("-" * 30)

print(f"Accuracy : {default_accuracy:.4f}")
print(f"Precision: {default_precision:.4f}")
print(f"Recall   : {default_recall:.4f}")
print(f"F1-score : {default_f1:.4f}")


print(f"\nProduction threshold: {threshold}")
print("-" * 30)

print(f"Accuracy : {threshold_accuracy:.4f}")
print(f"Precision: {threshold_precision:.4f}")
print(f"Recall   : {threshold_recall:.4f}")
print(f"F1-score : {threshold_f1:.4f}")


print("\nDefault confusion matrix:")
print(cm_default)

print("\nProduction threshold confusion matrix:")
print(cm_threshold)


# ============================================================
# SAVE METRICS CSV
# ============================================================

metrics = pd.DataFrame(
    [
        {
            "model": "Tuned K-Nearest Neighbors",
            "threshold_type": "default",
            "threshold": 0.50,
            "accuracy": default_accuracy,
            "precision": default_precision,
            "recall": default_recall,
            "f1_score": default_f1,
            "roc_auc": roc_auc,
            "test_samples": len(y_test),
        },
        {
            "model": "Tuned K-Nearest Neighbors",
            "threshold_type": "production",
            "threshold": threshold,
            "accuracy": threshold_accuracy,
            "precision": threshold_precision,
            "recall": threshold_recall,
            "f1_score": threshold_f1,
            "roc_auc": roc_auc,
            "test_samples": len(y_test),
        },
    ]
)

metrics_path = REPORTS_DIR / "evaluation_metrics.csv"

metrics.to_csv(
    metrics_path,
    index=False,
)

print(f"\nSaved: {metrics_path}")


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

cm_data = pd.DataFrame(
    {
        "actual_reject": [
            cm_threshold[0][0],
            cm_threshold[1][0],
        ],
        "actual_approve": [
            cm_threshold[0][1],
            cm_threshold[1][1],
        ],
    },
    index=[
        "predicted_reject",
        "predicted_approve",
    ],
)

cm_path = REPORTS_DIR / "confusion_matrix_values.csv"

cm_data.to_csv(cm_path)

print(f"Saved: {cm_path}")


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred_threshold,
    target_names=[
        "Reject",
        "Approve",
    ],
)

report_path = REPORTS_DIR / "classification_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "Credit Risk Engine - Classification Report\n"
    )

    file.write("=" * 55 + "\n\n")

    file.write(
        "Model: Tuned K-Nearest Neighbors\n"
    )

    file.write(
        f"Production threshold: {threshold}\n\n"
    )

    file.write(report)

print(f"Saved: {report_path}")


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_test,
    y_probability,
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"KNN (ROC-AUC = {roc_auc:.4f})",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Credit Risk Engine - ROC Curve"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

roc_path = REPORTS_DIR / "roc_curve.png"

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {roc_path}")


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_threshold,
    display_labels=[
        "Reject",
        "Approve",
    ],
)

disp.plot(
    cmap="Blues",
    values_format="d",
)

plt.title(
    f"Credit Risk Engine - Confusion Matrix\n"
    f"Threshold = {threshold}"
)

plt.tight_layout()

confusion_path = (
    REPORTS_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {confusion_path}")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 65)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 65)