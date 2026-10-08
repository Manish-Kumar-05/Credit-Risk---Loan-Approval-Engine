from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "loan_data.csv"
MODEL_PATH = BASE_DIR / "models" / "final_knn_pipeline.joblib"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

TARGET = "loan_status"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

X = X.copy()

X["loan_to_income"] = (
    X["loan_amnt"]
    / X["person_income"].replace(0, pd.NA)
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

print("Model loaded")


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print("Calculating permutation importance...")

result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

importance_df = pd.DataFrame(
    {
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }
)

importance_df = importance_df.sort_values(
    "importance_mean",
    ascending=False,
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = REPORTS_DIR / "feature_importance.csv"

importance_df.to_csv(
    csv_path,
    index=False,
)

print(f"Saved: {csv_path}")


# ============================================================
# TOP 10 FEATURES
# ============================================================

top_features = importance_df.head(10)

print("\nTop Features:")
print(
    top_features[
        [
            "feature",
            "importance_mean",
        ]
    ].to_string(index=False)
)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plot_df = top_features.sort_values(
    "importance_mean"
)

plt.barh(
    plot_df["feature"],
    plot_df["importance_mean"],
)

plt.xlabel(
    "Mean decrease in ROC-AUC"
)

plt.ylabel("Feature")

plt.title(
    "Credit Risk Engine - Feature Importance"
)

plt.tight_layout()


plot_path = (
    REPORTS_DIR /
    "feature_importance.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {plot_path}")

print("\nFeature importance completed successfully.")