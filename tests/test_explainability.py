import unittest
import os
import json
import numpy as np
from src.explainability import CreditRiskExplainer

# Monkeypatch SHAP to handle XGBoost 3.x base_score bracket format for testing
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

class TestProductionExplainabilityPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.model_path = "models/credit_risk_model.joblib"
        cls.metadata_path = "models/model_metadata.json"
        
        if not os.path.exists(cls.model_path) or not os.path.exists(cls.metadata_path):
            raise FileNotFoundError("Model and metadata must be persisted before running tests.")
            
        cls.explainer = CreditRiskExplainer(model_path=cls.model_path, metadata_path=cls.metadata_path)
        
        cls.valid_input = {
            "person_age": 30,
            "person_income": 60000,
            "person_home_ownership": "RENT",
            "person_emp_length": 5.0,
            "loan_intent": "PERSONAL",
            "loan_amnt": 10000,
            "loan_percent_income": 0.1667,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 5
        }
        
    def test_explainer_initialization(self):
        """Verify the CreditRiskExplainer class instantiates correctly with ensemble estimators."""
        self.assertEqual(len(self.explainer.explainers), 5)
        self.assertEqual(len(self.explainer.estimators), 5)
        self.assertIsNotNone(self.explainer.predictor)
        
    def test_feature_names_recovery(self):
        """Verify transformed feature names are recovered from fitted transformers."""
        features = self.explainer.transformed_features
        self.assertIn("person_age", features)
        self.assertIn("person_income", features)
        self.assertIn("person_home_ownership_RENT", features)
        
    def test_friendly_feature_mapping(self):
        """Verify technical column names are converted to friendly names and active categories."""
        # Numerical
        name, val = self.explainer.map_feature_to_friendly("person_income")
        self.assertEqual(name, "Annual Income")
        self.assertEqual(val, "")
        
        # Categorical
        name, val = self.explainer.map_feature_to_friendly("person_home_ownership_RENT")
        self.assertEqual(name, "Home Ownership Status")
        self.assertEqual(val, "RENT")
        
    def test_valid_explanation_generation(self):
        """Verify successful generation of local explanation structure."""
        res = self.explainer.explain(self.valid_input)
        self.assertTrue(res["success"])
        self.assertEqual(res["explanation_scope"], "underlying_xgboost_model_ensemble")
        self.assertIn("calibrated_risk_probability", res)
        self.assertIn("risk_category", res)
        self.assertIn("base_value", res)
        self.assertIn("top_risk_increasing_factors", res)
        self.assertIn("top_risk_decreasing_factors", res)
        
    def test_shap_additivity_verification(self):
        """Verify SHAP additivity properties holds true for underlying XGBoost models."""
        res = self.explainer.explain(self.valid_input)
        self.assertTrue(res["additivity_verified"])
        
    def test_factor_segregation_and_direction(self):
        """Verify risk increasing and decreasing factors are properly separated based on SHAP sign."""
        res = self.explainer.explain(self.valid_input)
        
        for factor in res["top_risk_increasing_factors"]:
            self.assertTrue(factor["shap_value"] > 0)
            self.assertEqual(factor["direction"], "increases_risk")
            
        for factor in res["top_risk_decreasing_factors"]:
            self.assertTrue(factor["shap_value"] < 0)
            self.assertEqual(factor["direction"], "decreases_risk")
            
    def test_sorting_by_absolute_magnitude(self):
        """Verify features are sorted in descending order of absolute contribution."""
        res = self.explainer.explain(self.valid_input)
        
        shaps_inc = [abs(f["shap_value"]) for f in res["top_risk_increasing_factors"]]
        shaps_dec = [abs(f["shap_value"]) for f in res["top_risk_decreasing_factors"]]
        
        self.assertEqual(shaps_inc, sorted(shaps_inc, reverse=True))
        self.assertEqual(shaps_dec, sorted(shaps_dec, reverse=True))
        
    def test_feature_exclusions_maintained(self):
        """Verify post-assessment columns remain excluded from feature mapping."""
        self.assertNotIn("loan_grade", self.explainer.transformed_features)
        self.assertNotIn("loan_int_rate", self.explainer.transformed_features)
        
    def test_predictor_probability_consistency(self):
        """Verify explanation probability score matches production predictor probability score."""
        res_exp = self.explainer.explain(self.valid_input)
        res_pred = self.explainer.predictor.predict(self.valid_input)
        
        self.assertEqual(res_exp["calibrated_risk_probability"], res_pred["risk_probability"])
        self.assertEqual(res_exp["risk_category"], res_pred["risk_category"])
        
    def test_invalid_input_rejections(self):
        """Verify invalid inputs are rejected and return error responses."""
        # Age out of range
        bad_input = self.valid_input.copy()
        bad_input["person_age"] = 12
        res = self.explainer.explain(bad_input)
        self.assertFalse(res["success"])
        self.assertIn("error", res)
        
        # NaN
        bad_input = self.valid_input.copy()
        bad_input["person_income"] = float("nan")
        res = self.explainer.explain(bad_input)
        self.assertFalse(res["success"])
        self.assertIn("NaN and Infinity are not allowed", res["error"])
        
        # Infinity
        bad_input = self.valid_input.copy()
        bad_input["loan_amnt"] = float("inf")
        res = self.explainer.explain(bad_input)
        self.assertFalse(res["success"])
        self.assertIn("NaN and Infinity are not allowed", res["error"])

if __name__ == "__main__":
    unittest.main()
