import sys
import pprint
from src.prediction import CreditPredictor

def run_verification():
    print("=== Production Artifact Verification ===")
    
    try:
        predictor = CreditPredictor()
        print("1. Deserialization Status: SUCCESS")
        print(f"2. Model Structure: {type(predictor.model)}")
        print(f"3. Expected Feature Count: {len(predictor.feature_names)} features")
        print(f"4. Expected Feature Names: {predictor.feature_names}")
        print(f"5. Classification Threshold: {predictor.classification_threshold}")
        print(f"6. Risk Category Boundaries: Low < {predictor.low_risk_upper}, High >= {predictor.high_risk_lower}")
        print("7. Artifact Compatibility: SUCCESS")
    except Exception as e:
        print(f"Verification setup failed: {e}")
        sys.exit(1)
        
    print("\n=== Applicant Verification Examples ===")
    
    # 1. Lower-risk applicant
    low_risk_input = {
        "person_age": 35,
        "person_income": 120000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 10.0,
        "loan_intent": "VENTURE",
        "loan_amnt": 10000,
        "loan_percent_income": 0.0833,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 8
    }
    
    # 2. Higher-risk applicant
    high_risk_input = {
        "person_age": 22,
        "person_income": 20000,
        "person_home_ownership": "RENT",
        "person_emp_length": 1.0,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_amnt": 15000,
        "loan_percent_income": 0.75,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 2
    }
    
    # 3. Borderline applicant
    borderline_input = {
        "person_age": 26,
        "person_income": 45000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "loan_intent": "MEDICAL",
        "loan_amnt": 12000,
        "loan_percent_income": 0.2667,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4
    }
    
    # 4. Missing field input
    missing_field_input = {
        "person_age": 25,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "loan_intent": "PERSONAL",
        # loan_amnt is missing
        "loan_percent_income": 0.1,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3
    }
    
    # 5. Invalid numeric input
    invalid_numeric_input = {
        "person_age": "twenty-five", # Should fail conversion
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "loan_intent": "PERSONAL",
        "loan_amnt": 5000,
        "loan_percent_income": 0.1,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3
    }
    
    # 6. NaN input
    nan_input = {
        "person_age": 25,
        "person_income": float("nan"), # NaN
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "loan_intent": "PERSONAL",
        "loan_amnt": 5000,
        "loan_percent_income": 0.1,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3
    }
    
    # 7. Infinity input
    inf_input = {
        "person_age": 25,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "loan_intent": "PERSONAL",
        "loan_amnt": float("inf"), # Infinity
        "loan_percent_income": 0.1,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3
    }
    
    # 8. Invalid categorical input
    invalid_cat_input = {
        "person_age": 25,
        "person_income": 50000,
        "person_home_ownership": "JUNK_OWNERSHIP", # Invalid category
        "person_emp_length": 3.0,
        "loan_intent": "PERSONAL",
        "loan_amnt": 5000,
        "loan_percent_income": 0.1,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3
    }
    
    # 9. Logical inconsistency: emp_length > age
    impossible_emp_input = {
        "person_age": 20,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 15.0, # working since age 5 (implausible full-time)
        "loan_intent": "PERSONAL",
        "loan_amnt": 5000,
        "loan_percent_income": 0.1,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3
    }

    test_cases = [
        ("Lower-Risk Applicant", low_risk_input),
        ("Higher-Risk Applicant", high_risk_input),
        ("Borderline Applicant", borderline_input),
        ("Missing Field Input", missing_field_input),
        ("Invalid Numeric Input", invalid_numeric_input),
        ("NaN Input", nan_input),
        ("Infinity Input", inf_input),
        ("Invalid Categorical Input", invalid_cat_input),
        ("Logical Inconsistency (Emp Length vs Age)", impossible_emp_input)
    ]
    
    for label, payload in test_cases:
        print(f"\n--- Test: {label} ---")
        output = predictor.predict(payload)
        pprint.pprint(output)

if __name__ == "__main__":
    run_verification()
