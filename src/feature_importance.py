from pathlib import Path

import joblib
import pandas as pd

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

OUTPUT_PATH = (
    BASE_DIR
    / "models"
    / "feature_importance.csv"
)


# ==================================================
# Calculate permutation importance
# ==================================================

def calculate_feature_importance():

    print("Loading trained model...")

    model = joblib.load(MODEL_PATH)

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------

    df["loan_to_income"] = (
        df["loan_amnt"]
        / df["person_income"]
    )

    # --------------------------------------------------
    # Separate features and target
    # --------------------------------------------------

    X = df.drop(
        columns=["loan_status"]
    )

    y = df["loan_status"]

    # --------------------------------------------------
    # Train/test split
    # --------------------------------------------------

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Calculating permutation importance...")

    # --------------------------------------------------
    # Permutation importance
    # --------------------------------------------------

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
    )

    print("Permutation importance completed.")

    # --------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    # --------------------------------------------------
    # Sort
    # --------------------------------------------------

    importance_df = (
        importance_df
        .sort_values(
            by="importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importance_df


# ==================================================
# Save feature importance
# ==================================================

def save_feature_importance():

    importance_df = calculate_feature_importance()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nPermutation Feature Importance")
    print("-" * 60)

    print(
        importance_df.to_string(
            index=False
        )
    )

    print("\nSaved to:")

    print(OUTPUT_PATH)


# ==================================================
# Run only when executed directly
# ==================================================

if __name__ == "__main__":

    save_feature_importance()