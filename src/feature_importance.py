print("feature_importance.py started")

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "final_knn_pipeline.joblib"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "loan_data.csv"
)


# ==================================================
# Load trained model
# ==================================================

print("Loading model...")
model = joblib.load(MODEL_PATH)


# ==================================================
# Load dataset
# ==================================================

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)


# ==================================================
# Feature engineering
# ==================================================

df["loan_to_income"] = (
    df["loan_amnt"]
    / df["person_income"]
)


# ==================================================
# Separate features and target
# ==================================================

X = df.drop(
    columns=["loan_status"]
)

y = df["loan_status"]


# ==================================================
# Train / test split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ==================================================
# Permutation importance
# ==================================================

print("Starting permutation importance...")
result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
)


# ==================================================
# Create importance DataFrame
# ==================================================

importance_df = pd.DataFrame(
    {
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }
)


# ==================================================
# Sort features by importance
# ==================================================

importance_df = (
    importance_df
    .sort_values(
        by="importance_mean",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ==================================================
# Display results in terminal
# ==================================================

print("\nPermutation Feature Importance")
print("-" * 50)

print(
    importance_df.to_string(
        index=False
    )
)


# ==================================================
# Plot top 10 features
# ==================================================

top_features = importance_df.head(10)


plt.figure(figsize=(12, 8))

plt.barh(
    top_features["feature"],
    top_features["importance_mean"],
)

plt.xlabel(
    "Mean decrease in ROC-AUC"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 10 Features - Permutation Importance"
)

# Highest importance at the top
plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# ==================================================
# Function for Streamlit
# ==================================================

def get_feature_importance():
    """
    Return permutation feature importance
    as a DataFrame.
    """

    return importance_df

print("Permutation importance completed")