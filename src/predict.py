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
# Validate applicant
# --------------------------------------------------

def validate_applicant(applicant_data):
    """Validate applicant input before prediction."""

    # ----------------------------------------------
    # Check required fields
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
    # Numeric validation
    # ----------------------------------------------

    if not 18 <= applicant_data["person_age"] <= 100:
        raise ValueError(
            "person_age must be between 18 and 100."
        )

    if applicant_data["person_income"] <= 0:
        raise ValueError(
            "person_income must be greater than 0."
        )

    if applicant_data["person_emp_exp"] < 0:
        raise ValueError(
            "person_emp_exp cannot be negative."
        )

    if applicant_data["loan_amnt"] <= 0:
        raise ValueError(
            "loan_amnt must be greater than 0."
        )

    if not 0 <= applicant_data["loan_percent_income"] <= 1:
        raise ValueError(
            "loan_percent_income must be between 0 and 1."
        )

    if not 0 <= applicant_data["loan_int_rate"] <= 100:
        raise ValueError(
            "loan_int_rate must be between 0 and 100."
        )

    if applicant_data["cb_person_cred_hist_length"] < 0:
        raise ValueError(
            "Credit history length cannot be negative."
        )

    if not 300 <= applicant_data["credit_score"] <= 850:
        raise ValueError(
            "credit_score must be between 300 and 850."
        )

    # ----------------------------------------------
    # Categorical validation
    # ----------------------------------------------

    allowed_values = {
        "person_gender": {
            "male",
            "female",
        },

        "person_education": {
            "High School",
            "Bachelor",
            "Master",
            "PhD",
        },

        "person_home_ownership": {
            "RENT",
            "OWN",
            "MORTGAGE",
            "OTHER",
        },

        "loan_intent": {
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION",
        },

        "previous_loan_defaults_on_file": {
            "Yes",
            "No",
        },
    }

    for column, valid_values in allowed_values.items():

        value = applicant_data[column]

        if value not in valid_values:
            raise ValueError(
                f"Invalid value for {column}: {value}. "
                f"Allowed values: {sorted(valid_values)}"
            )


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
    # Validate applicant
    # ----------------------------------------------

    validate_applicant(applicant_data)

    # ----------------------------------------------
    # Create DataFrame
    # ----------------------------------------------

    applicant_df = pd.DataFrame(
        [applicant_data]
    )

    # ----------------------------------------------
    # Feature engineering
    # ----------------------------------------------

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
            4,
        ),
        "prediction": prediction,
        "decision": decision,
        "risk_category": risk_category,
        "threshold": threshold,
    }


# Low Risk here means high predicted approval probability.