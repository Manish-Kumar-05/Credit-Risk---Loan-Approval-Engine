from predict import predict_loan


applicant = {
    "person_age": 22,
    "person_gender": "male",
    "person_education": "High School",
    "person_income": 150000,
    "person_emp_exp": 10,
    "person_home_ownership": "RENT",
    "loan_amnt": 12000,
    "loan_intent": "PERSONAL",
    "loan_int_rate": 18.5,
    "loan_percent_income": 0.80,
    "cb_person_cred_hist_length": 2,
    "credit_score": 520,
    "previous_loan_defaults_on_file": "No",
}


result = predict_loan(applicant)


print("\nLoan Application Result")
print("-" * 30)

for key, value in result.items():
    print(f"{key}: {value}")