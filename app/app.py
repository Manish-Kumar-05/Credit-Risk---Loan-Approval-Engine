import sys
from pathlib import Path

import streamlit as st


# --------------------------------------------------
# Project path
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))


# --------------------------------------------------
# Import prediction engine
# --------------------------------------------------

from src.predict import predict_loan


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Engine",
    page_icon="💳",
    layout="wide",
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    'Credit Risk & Loan Approval Engine'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Machine learning based loan approval prediction'
    '</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Applicant information
# --------------------------------------------------

st.header("Applicant Information")

col1, col2, col3 = st.columns(3)


with col1:

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25,
    )

    person_gender = st.selectbox(
        "Gender",
        [
            "male",
            "female",
        ],
    )

    person_education = st.selectbox(
        "Education",
        [
            "High School",
            "Bachelor",
            "Master",
            "PhD",
        ],
    )


with col2:

    person_income = st.number_input(
        "Annual Income",
        min_value=1.0,
        value=50000.0,
        step=1000.0,
    )

    person_emp_exp = st.number_input(
        "Employment Experience (years)",
        min_value=0,
        max_value=100,
        value=3,
    )

    person_home_ownership = st.selectbox(
        "Home Ownership",
        [
            "RENT",
            "OWN",
            "MORTGAGE",
            "OTHER",
        ],
    )


with col3:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=700,
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (years)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
    )

    previous_loan_defaults_on_file = st.selectbox(
        "Previous Loan Defaults",
        [
            "No",
            "Yes",
        ],
    )


# --------------------------------------------------
# Loan information
# --------------------------------------------------

st.header("Loan Information")

col1, col2, col3 = st.columns(3)


with col1:

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=1.0,
        value=10000.0,
        step=500.0,
    )

    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION",
        ],
    )


with col2:

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.5,
        step=0.1,
    )

    loan_percent_income = st.number_input(
        "Loan / Income Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
    )


# --------------------------------------------------
# Prediction button
# --------------------------------------------------

st.divider()

predict_button = st.button(
    "Predict Loan Approval",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    applicant = {
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership":
            person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income":
            loan_percent_income,
        "cb_person_cred_hist_length":
            cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file":
            previous_loan_defaults_on_file,
    }

    try:

        result = predict_loan(applicant)

        st.divider()

        st.header("Prediction Result")

        # ------------------------------------------
        # Metrics
        # ------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Approval Probability",
                f"{result['approval_probability'] * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Decision",
                result["decision"]
            )

        with col3:

            st.metric(
                "Risk Category",
                result["risk_category"]
            )

        # ------------------------------------------
        # Decision message
        # ------------------------------------------

        if result["decision"] == "Approve":

            st.success(
                "The model predicts that this "
                "application should be approved."
            )

        else:

            st.error(
                "The model predicts that this "
                "application should be rejected."
            )


    except ValueError as error:

        st.error(str(error))