from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_knn_pipeline.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "decision_threshold.joblib"

ROC_PATH = BASE_DIR / "reports" / "roc_curve.png"
CONFUSION_PATH = BASE_DIR / "reports" / "confusion_matrix.png"
METRICS_PATH = BASE_DIR / "reports" / "evaluation_metrics.csv"
FEATURE_IMPORTANCE_PATH = (
    BASE_DIR / "reports" / "feature_importance.csv"
)

FEATURE_IMPORTANCE_IMAGE = (
    BASE_DIR / "reports" / "feature_importance.png"
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        background-color: rgba(128,128,128,0.05);
        text-align: center;
    }

    .decision-box {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.3);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .section-title {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .small-text {
        font-size: 0.85rem;
        opacity: 0.75;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_threshold():
    return joblib.load(THRESHOLD_PATH)


model = load_model()
threshold = load_threshold()


# ============================================================
# HEADER
# ============================================================

st.title("Credit Risk Engine")

st.markdown(
    """
    **ML-powered loan approval assessment**

    Enter applicant information to estimate the probability
    of loan approval and classify the applicant's risk level.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model Information")

    st.write("**Model:** Tuned K-Nearest Neighbors")

    st.write(
        f"**Decision threshold:** {threshold:.2f}"
    )

    st.write("**Target:** Loan approval")

    st.write("**0:** Reject")

    st.write("**1:** Approve")

    st.divider()

    st.subheader("How it works")

    st.write(
        """
        1. Enter applicant information.
        2. The trained ML pipeline processes the data.
        3. The model calculates approval probability.
        4. The probability is compared with the
           production threshold.
        5. The application is classified as
           Approve or Reject.
        """
    )

    st.divider()

    st.caption(
        "Credit Risk Engine — Machine Learning Project"
    )


# ============================================================
# INPUT SECTIONS
# ============================================================

st.header("Applicant Information")


# ------------------------------------------------------------
# PERSONAL INFORMATION
# ------------------------------------------------------------

st.subheader("Personal Information")

col1, col2, col3 = st.columns(3)

with col1:

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25,
        step=1,
    )

with col2:

    person_gender = st.selectbox(
        "Gender",
        [
            "male",
            "female",
        ],
    )

with col3:

    person_education = st.selectbox(
        "Education",
        [
            "High School",
            "Associate",
            "Bachelor",
            "Master",
            "Doctorate",
        ],
    )


# ------------------------------------------------------------
# FINANCIAL INFORMATION
# ------------------------------------------------------------

st.subheader("Financial Information")

col1, col2, col3 = st.columns(3)

with col1:

    person_income = st.number_input(
        "Annual Income",
        min_value=0,
        value=60000,
        step=1000,
    )

with col2:

    person_emp_exp = st.number_input(
        "Employment Experience (years)",
        min_value=0,
        max_value=60,
        value=3,
        step=1,
    )

with col3:

    person_home_ownership = st.selectbox(
        "Home Ownership",
        [
            "RENT",
            "MORTGAGE",
            "OWN",
            "OTHER",
        ],
    )


# ------------------------------------------------------------
# LOAN INFORMATION
# ------------------------------------------------------------

st.subheader("Loan Information")

col1, col2, col3 = st.columns(3)

with col1:

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0,
        value=10000,
        step=500,
    )

with col2:

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=12.0,
        step=0.1,
    )

with col3:

    loan_percent_income = st.number_input(
        "Loan / Income Ratio",
        min_value=0.0,
        max_value=2.0,
        value=0.20,
        step=0.01,
    )


col1, col2 = st.columns(2)

with col1:

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

    previous_loan_defaults_on_file = st.selectbox(
        "Previous Loan Defaults",
        [
            "No",
            "Yes",
        ],
    )


# ------------------------------------------------------------
# CREDIT HISTORY
# ------------------------------------------------------------

st.subheader("Credit History")

col1, col2 = st.columns(2)

with col1:

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (years)",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
    )

with col2:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1,
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "Assess Credit Risk",
    type="primary",
    use_container_width=True,
)


if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            {
                "person_age": person_age,
                "person_gender": person_gender,
                "person_education": person_education,
                "person_income": person_income,
                "person_emp_exp": person_emp_exp,
                "person_home_ownership": person_home_ownership,
                "loan_amnt": loan_amnt,
                "loan_intent": loan_intent,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_cred_hist_length":
                    cb_person_cred_hist_length,
                "credit_score": credit_score,
                "previous_loan_defaults_on_file":
                    previous_loan_defaults_on_file,
            }
        ]
    )


    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    input_data["loan_to_income"] = (
        input_data["loan_amnt"]
        / input_data["person_income"].replace(0, pd.NA)
    )

    input_data["loan_to_income"] = (
        input_data["loan_to_income"].fillna(0)
    )


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_data
    )[0][1]


    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    prediction = int(
        probability >= threshold
    )


    if prediction == 1:

        decision = "Approve"

    else:

        decision = "Reject"


    # --------------------------------------------------------
    # RISK CATEGORY
    # --------------------------------------------------------

    if probability >= 0.70:

        risk_category = "Low Risk"

    elif probability >= 0.40:

        risk_category = "Medium Risk"

    else:

        risk_category = "High Risk"


        # --------------------------------------------------------
    # APPLICANT EXPLANATION
    # --------------------------------------------------------

    st.subheader("Applicant Explanation")

    explanation_data = {
        "Credit Score": credit_score,
        "Annual Income": person_income,
        "Loan Amount": loan_amnt,
        "Interest Rate": loan_int_rate,
        "Loan / Income Ratio": loan_percent_income,
        "Credit History Length": cb_person_cred_hist_length,
        "Employment Experience": person_emp_exp,
    }

    explanation_df = pd.DataFrame(
        list(explanation_data.items()),
        columns=["Feature", "Applicant Value"],
    )


    # --------------------------------------------------------
    # SIMPLE RISK SIGNALS
    # --------------------------------------------------------

    signals = []


    # Credit score

    if credit_score >= 700:

        signals.append(
            (
                "Credit Score",
                "Positive",
                "Strong credit score",
            )
        )

    elif credit_score < 600:

        signals.append(
            (
                "Credit Score",
                "Negative",
                "Low credit score",
            )
        )

    else:

        signals.append(
            (
                "Credit Score",
                "Neutral",
                "Moderate credit score",
            )
        )


    # Loan / income

    if loan_percent_income <= 0.20:

        signals.append(
            (
                "Loan / Income Ratio",
                "Positive",
                "Loan burden is relatively low",
            )
        )

    elif loan_percent_income > 0.50:

        signals.append(
            (
                "Loan / Income Ratio",
                "Negative",
                "Loan burden is relatively high",
            )
        )

    else:

        signals.append(
            (
                "Loan / Income Ratio",
                "Neutral",
                "Moderate loan burden",
            )
        )


    # Interest rate

    if loan_int_rate <= 10:

        signals.append(
            (
                "Interest Rate",
                "Positive",
                "Relatively low interest rate",
            )
        )

    elif loan_int_rate >= 18:

        signals.append(
            (
                "Interest Rate",
                "Negative",
                "High interest rate",
            )
        )

    else:

        signals.append(
            (
                "Interest Rate",
                "Neutral",
                "Moderate interest rate",
            )
        )


    # Previous defaults

    if previous_loan_defaults_on_file == "Yes":

        signals.append(
            (
                "Previous Defaults",
                "Negative",
                "Previous loan default recorded",
            )
        )

    else:

        signals.append(
            (
                "Previous Defaults",
                "Positive",
                "No previous loan default recorded",
            )
        )


    # Income

    if person_income >= 100000:

        signals.append(
            (
                "Income",
                "Positive",
                "Relatively high annual income",
            )
        )

    elif person_income < 30000:

        signals.append(
            (
                "Income",
                "Negative",
                "Relatively low annual income",
            )
        )

    else:

        signals.append(
            (
                "Income",
                "Neutral",
                "Moderate annual income",
            )
        )


    # --------------------------------------------------------
    # DISPLAY SIGNALS
    # --------------------------------------------------------

    signal_df = pd.DataFrame(
        signals,
        columns=[
            "Feature",
            "Signal",
            "Explanation",
        ],
    )

    st.dataframe(
        signal_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.header("Assessment Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Approval Probability",
            f"{probability:.2%}",
        )

    with col2:

        st.metric(
            "Decision",
            decision,
        )

    with col3:

        st.metric(
            "Risk Category",
            risk_category,
        )


    # --------------------------------------------------------
    # DECISION MESSAGE
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="decision-box">

        <h3>Assessment: {decision}</h3>

        <p>
        The model estimates an approval probability of
        <strong>{probability:.2%}</strong>.
        </p>

        <p>
        The production decision threshold is
        <strong>{threshold:.2f}</strong>.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------------

    st.subheader("Approval Probability")

    st.progress(
        min(max(probability, 0.0), 1.0)
    )


    # --------------------------------------------------------
    # INPUT SUMMARY
    # --------------------------------------------------------

    with st.expander(
        "View Applicant Data"
    ):

        st.dataframe(
            input_data,
            use_container_width=True,
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header("Model Performance")


if METRICS_PATH.exists():

    metrics_df = pd.read_csv(
        METRICS_PATH
    )

    production_metrics = metrics_df[
        metrics_df["threshold_type"] == "production"
    ]

    if not production_metrics.empty:

        metrics = production_metrics.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Accuracy",
                f"{metrics['accuracy']:.2%}",
            )

        with col2:

            st.metric(
                "Precision",
                f"{metrics['precision']:.2%}",
            )

        with col3:

            st.metric(
                "Recall",
                f"{metrics['recall']:.2%}",
            )

        with col4:

            st.metric(
                "ROC-AUC",
                f"{metrics['roc_auc']:.2%}",
            )


# ============================================================
# EVALUATION VISUALIZATIONS
# ============================================================

st.subheader("Evaluation Visualizations")

col1, col2 = st.columns(2)

with col1:

    if CONFUSION_PATH.exists():

        st.image(
            str(CONFUSION_PATH),
            caption="Confusion Matrix",
            use_container_width=True,
        )

with col2:

    if ROC_PATH.exists():

        st.image(
            str(ROC_PATH),
            caption="ROC Curve",
            use_container_width=True,
        )

# ============================================================
# MODEL EXPLANATION
# ============================================================

st.divider()

st.header("Model Explanation")

st.markdown(
    """
    The model uses multiple applicant characteristics to estimate
    loan approval probability.

    Feature importance below is calculated using permutation
    importance. A larger value means that shuffling that feature
    caused a larger decrease in model performance.
    """
)


if FEATURE_IMPORTANCE_IMAGE.exists():

    st.image(
        str(FEATURE_IMPORTANCE_IMAGE),
        caption="Global Feature Importance",
        use_container_width=True,
    )


if FEATURE_IMPORTANCE_PATH.exists():

    importance_df = pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )

    st.subheader("Most Important Features")

    st.dataframe(
        importance_df.head(10),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "This application is a machine learning demonstration "
    "and should not be used as the sole basis for real-world "
    "financial decisions."
)