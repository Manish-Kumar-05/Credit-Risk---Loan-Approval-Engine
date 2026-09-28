from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "final_knn_pipeline.joblib"
)

THRESHOLD_PATH = (
    BASE_DIR
    / "models"
    / "decision_threshold.joblib"
)


# --------------------------------------------------
# Load trained model and threshold
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)


# --------------------------------------------------
# Required input columns
# --------------------------------------------------

REQUIRED_COLUMNS = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
]


# --------------------------------------------------
# Risk category
# --------------------------------------------------

def get_risk_category(probability):
    """
    Categorize applicant based on approval probability.

    Note:
    This is an application-level category based on
    approval probability, not actual default probability.
    """

    if probability >= 0.75:
        return "Low Risk"

    elif probability >= threshold:
        return "Medium Risk"

    else:
        return "High Risk"


# --------------------------------------------------
# Predict loan application
# --------------------------------------------------

def predict_loan(applicant_data):
    """
    Predict whether a loan application should be
    approved or rejected.

    Parameters
    ----------
    applicant_data : dict
        Applicant information.

    Returns
    -------
    dict
        Prediction result.
    """

    # ----------------------------------------------
    # Check required columns
    # ----------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in applicant_data
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required fields: {missing_columns}"
        )

    # ----------------------------------------------
    # Create DataFrame
    # ----------------------------------------------

    applicant_df = pd.DataFrame(
        [applicant_data]
    )

    # ----------------------------------------------
    # Feature engineering
    # ----------------------------------------------

    if applicant_df["person_income"].iloc[0] <= 0:
        raise ValueError(
            "person_income must be greater than 0."
        )

    applicant_df["loan_to_income"] = (
        applicant_df["loan_amnt"]
        / applicant_df["person_income"]
    )

    # ----------------------------------------------
    # Get approval probability
    # ----------------------------------------------

    approval_probability = (
        model.predict_proba(applicant_df)[0, 1]
    )

    # ----------------------------------------------
    # Apply threshold
    # ----------------------------------------------

    prediction = int(
        approval_probability >= threshold
    )

    # ----------------------------------------------
    # Convert prediction to decision
    # ----------------------------------------------

    decision = (
        "Approve"
        if prediction == 1
        else "Reject"
    )

    # ----------------------------------------------
    # Risk category
    # ----------------------------------------------

    risk_category = get_risk_category(
        approval_probability
    )

    # ----------------------------------------------
    # Return result
    # ----------------------------------------------

    return {
        "approval_probability": round(
            float(approval_probability),
            4
        ),
        "prediction": prediction,
        "decision": decision,
        "risk_category": risk_category,
        "threshold": threshold,
    }


# --------------------------------------------------
# Test prediction
# --------------------------------------------------

if __name__ == "__main__":

    applicant = {
        "person_age": 25,
        "person_gender": "male",
        "person_education": "Bachelor",
        "person_income": 50000,
        "person_emp_exp": 3,
        "person_home_ownership": "RENT",
        "loan_amnt": 10000,
        "loan_intent": "EDUCATION",
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.20,
        "cb_person_cred_hist_length": 5,
        "credit_score": 700,
        "previous_loan_defaults_on_file": "No",
    }

    result = predict_loan(applicant)

    print("\nLoan Application Result")
    print("-" * 30)

    for key, value in result.items():
        print(f"{key}: {value}")