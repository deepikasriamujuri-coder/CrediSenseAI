import time
import pprint
from src.explainability import CreditRiskExplainer

# Monkeypatch SHAP to handle XGBoost 3.x base_score bracket format (in case running directly)
import shap.explainers._tree as shap_tree
from shap.explainers.other._ubjson import decode_ubjson_buffer

_original_decode_ubjson = decode_ubjson_buffer

def _patched_decode_ubjson(*args, **kwargs):
    result = _original_decode_ubjson(*args, **kwargs)
    try:
        if "learner" in result:
            learner = result["learner"]
            if "learner_model_param" in learner:
                params = learner["learner_model_param"]
                if "base_score" in params:
                    val = str(params["base_score"])
                    if val.startswith("[") and val.endswith("]"):
                        params["base_score"] = val[1:-1]
    except Exception:
        pass
    return result

shap_tree.decode_ubjson_buffer = _patched_decode_ubjson

def run_verification():
    print("=== Reusable CreditRiskExplainer Verification ===")
    
    start_init = time.time()
    explainer = CreditRiskExplainer()
    init_time = time.time() - start_init
    print(f"Explainer Class Initialization Latency: {init_time:.4f} seconds")
    
    # Payload examples
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
    
    test_cases = [
        ("Lower-Risk Applicant Case", low_risk_input),
        ("Higher-Risk Applicant Case", high_risk_input),
        ("Borderline Applicant Case", borderline_input)
    ]
    
    for label, payload in test_cases:
        print(f"\n--- Testing: {label} ---")
        
        # Measure latency
        start_exp = time.time()
        explanation = explainer.explain(payload)
        exp_time = time.time() - start_exp
        
        print(f"Explanation Generation Latency: {exp_time:.4f} seconds")
        print("Calibrated Risk Probability:", explanation.get("calibrated_risk_probability"))
        print("Risk Category              :", explanation.get("risk_category"))
        print("SHAP Additivity Verified   :", explanation.get("additivity_verified"))
        print("Base Value (Avg log-odds)  :", explanation.get("base_value"))
        
        print("\nTop Risk-Increasing Factors:")
        for factor in explanation.get("top_risk_increasing_factors", []):
            print(f"  - {factor['friendly_feature']}: {factor['value']} (SHAP: {factor['shap_value']:+})")
            print(f"    Text: {factor['description']}")
            
        print("\nTop Risk-Decreasing Factors:")
        for factor in explanation.get("top_risk_decreasing_factors", []):
            print(f"  - {factor['friendly_feature']}: {factor['value']} (SHAP: {factor['shap_value']:+})")
            print(f"    Text: {factor['description']}")
            
        print("\nFull JSON Response Snippet:")
        # Display only first 2 items of increasing/decreasing in pprint to prevent logs cluttering
        explanation_snippet = explanation.copy()
        explanation_snippet["top_risk_increasing_factors"] = explanation_snippet["top_risk_increasing_factors"][:2]
        explanation_snippet["top_risk_decreasing_factors"] = explanation_snippet["top_risk_decreasing_factors"][:2]
        pprint.pprint(explanation_snippet)
        
    # Latency study
    print("\n=== Explanation Performance Study ===")
    latencies = []
    for _ in range(20):
        t0 = time.time()
        _ = explainer.explain(low_risk_input)
        latencies.append(time.time() - t0)
        
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average explanation generation latency across 20 iterations: {avg_latency:.4f} seconds")
    print(f"Single explanation latency range: {min(latencies):.4f}s to {max(latencies):.4f}s")
    
    if avg_latency > 0.10:
        print("Recommendation: Explainability queries should be generated on-demand or cached since latency is >100ms.")
    else:
        print("Recommendation: Explainability queries can be generated synchronously per web request since latency is <100ms.")

if __name__ == "__main__":
    run_verification()
