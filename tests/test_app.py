import unittest
import json
import os
from io import BytesIO

# Apply SHAP monkeypatch in test process
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

# Import app.py globals to mock states
import app as flask_app

class TestFlaskEndpoints(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create Flask test client under testing config
        cls.app = flask_app.create_app("testing")
        cls.client = cls.app.test_client()
        
        # Valid test applicant data payloads
        cls.low_risk_payload = {
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
        
        cls.high_risk_payload = {
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
        
        cls.borderline_payload = {
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
        
    def test_browser_get_routes(self):
        """Verify that basic browser routes load successfully (HTTP 200)."""
        # GET /
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Risk Assessment", res.data)
        
        # GET /about
        res = self.client.get("/about")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"System Architecture & Methodology", res.data)
        self.assertIn(b"Pipeline Flow", res.data)
        
    def test_health_endpoint(self):
        """Verify health check endpoint reports status and codes correctly."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["prediction_service_available"])
        
    def test_api_model_info(self):
        """Verify model metadata API route returns correct metadata and structure."""
        res = self.client.get("/api/model-info")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data["success"])
        self.assertIn("model_info", data)
        self.assertEqual(data["model_info"]["calibration_method"], "isotonic")
        self.assertNotIn("C:\\", res.text)  # No local file paths
        
    def test_api_predict_success_cases(self):
        """Verify POST /api/predict correctly handles low, high, and borderline inputs."""
        # Low risk
        res = self.client.post("/api/predict", json=self.low_risk_payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["prediction"]["risk_category"], "Low")
        
        # High risk
        res = self.client.post("/api/predict", json=self.high_risk_payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["prediction"]["risk_category"], "High")
        
        # Borderline (predicted low risk under 0.50 threshold but validates category logic)
        res = self.client.post("/api/predict", json=self.borderline_payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["prediction"]["risk_category"], "Low")
        
    def test_api_explain_success_case(self):
        """Verify POST /api/explain returns a valid SHAP explanation response."""
        res = self.client.post("/api/explain", json=self.low_risk_payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data["success"])
        if data["explanation_available"]:
            self.assertIn("explanation", data)
            self.assertIn("base_value", data["explanation"])
            
    def test_api_predict_with_explanation_success_case(self):
        """Verify combined prediction & explanation endpoint returns structured output."""
        res = self.client.post("/api/predict-with-explanation", json=self.low_risk_payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data["success"])
        self.assertIn("prediction", data)
        if data["explanation_available"]:
            self.assertIsNotNone(data["explanation"])
            
    def test_content_type_handling(self):
        """Verify API endpoints reject payloads that are not application/json."""
        res = self.client.post("/api/predict", data="plain text payload")
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertFalse(data["success"])
        self.assertEqual(data["error"]["type"], "validation_error")
        
    def test_missing_json_body(self):
        """Verify API endpoints reject empty json requests."""
        res = self.client.post("/api/predict", headers={"Content-Type": "application/json"})
        self.assertEqual(res.status_code, 400)
        
    def test_missing_field_rejection(self):
        """Verify missing fields trigger a validation error."""
        bad_payload = self.low_risk_payload.copy()
        bad_payload.pop("loan_amnt")
        res = self.client.post("/api/predict", json=bad_payload)
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertIn("Missing required field", data["error"]["message"])
        
    def test_non_numeric_input_rejection(self):
        """Verify string values in numerical fields trigger a validation error."""
        bad_payload = self.low_risk_payload.copy()
        bad_payload["person_age"] = "thirty-five"
        res = self.client.post("/api/predict", json=bad_payload)
        self.assertEqual(res.status_code, 400)
        
    def test_nan_rejection(self):
        """Verify NaN values trigger validation error."""
        bad_payload = self.low_risk_payload.copy()
        bad_payload["person_income"] = float("nan")
        res = self.client.post("/api/predict", json=bad_payload)
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertIn("NaN and Infinity are not allowed", data["error"]["message"])
        
    def test_infinity_rejection(self):
        """Verify Infinity values trigger validation error."""
        bad_payload = self.low_risk_payload.copy()
        bad_payload["loan_amnt"] = float("inf")
        res = self.client.post("/api/predict", json=bad_payload)
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertIn("NaN and Infinity are not allowed", data["error"]["message"])
        
    def test_invalid_category_rejection(self):
        """Verify invalid categorical categories trigger validation error."""
        bad_payload = self.low_risk_payload.copy()
        bad_payload["person_home_ownership"] = "COOP_JUNK"
        res = self.client.post("/api/predict", json=bad_payload)
        self.assertEqual(res.status_code, 400)
        
    def test_out_of_range_value_rejection(self):
        """Verify out of range age boundary triggers validation error."""
        bad_payload = self.low_risk_payload.copy()
        bad_payload["person_age"] = 15  # age < 18
        res = self.client.post("/api/predict", json=bad_payload)
        self.assertEqual(res.status_code, 400)
        
    def test_logical_inconsistency_rejection(self):
        """Verify impossible employment length vs age triggers validation error."""
        bad_payload = self.low_risk_payload.copy()
        bad_payload["person_age"] = 25
        bad_payload["person_emp_length"] = 20.0  # working since age 5
        res = self.client.post("/api/predict", json=bad_payload)
        self.assertEqual(res.status_code, 400)
        
    def test_unknown_route(self):
        """Verify accessing an unknown endpoint returns 404."""
        res = self.client.get("/api/unknown-endpoint-path")
        self.assertEqual(res.status_code, 404)
        
    def test_unsupported_http_method(self):
        """Verify calling GET on a POST-only endpoint returns 405."""
        res = self.client.get("/api/predict")
        self.assertEqual(res.status_code, 405)
        
    def test_oversized_request(self):
        """Verify large requests (larger than config limit) return 413."""
        # Create a payload of 3MB (config max is 2MB)
        large_payload = "X" * (3 * 1024 * 1024)
        res = self.client.post("/api/predict", data=large_payload, content_type="application/json")
        self.assertEqual(res.status_code, 413)
        
    def test_prediction_service_unavailable(self):
        """Verify that when PREDICTION_AVAILABLE is False, request fails with 503."""
        flask_app.PREDICTION_AVAILABLE = False
        try:
            res = self.client.post("/api/predict", json=self.low_risk_payload)
            self.assertEqual(res.status_code, 503)
            data = json.loads(res.data)
            self.assertEqual(data["error"]["type"], "service_unavailable")
        finally:
            # Restore state
            flask_app.PREDICTION_AVAILABLE = True
            
    def test_shap_service_unavailable_degradation(self):
        """Verify explanation requests degrade gracefully with success: True if SHAP fails."""
        flask_app.SHAP_AVAILABLE = False
        try:
            res = self.client.post("/api/explain", json=self.low_risk_payload)
            self.assertEqual(res.status_code, 200)
            data = json.loads(res.data)
            self.assertTrue(data["success"])
            self.assertFalse(data["explanation_available"])
            self.assertIn("SHAP explanations are currently unavailable", data["message"])
        finally:
            flask_app.SHAP_AVAILABLE = True
            
    def test_html_form_submission_success(self):
        """Verify that a valid HTML form submission returns success template."""
        res = self.client.post("/predict", data=self.low_risk_payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Assessment Dashboard", res.data)
        self.assertIn(b"Visual Risk Gauge", res.data)
        self.assertIn(b"Top Risk-Decreasing Factors", res.data)
        
    def test_html_form_submission_invalid(self):
        """Verify that an invalid HTML form submission returns bad request error page."""
        bad_payload = self.low_risk_payload.copy()
        bad_payload["person_age"] = 10
        res = self.client.post("/predict", data=bad_payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Validation Error", res.data)
        self.assertIn(b"Validation Tips", res.data)

    def test_premium_homepage_elements(self):
        """Verify presence of hero section, metrics, workflow, form fields and footer on homepage."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Understand Credit Risk", res.data)  # Hero Title
        self.assertIn(b"AI-Powered Credit Analytics", res.data)  # Hero Badge
        self.assertIn(b"Tuned Ensemble Model", res.data)  # Metrics
        self.assertIn(b"System Workflow Steps", res.data)  # Workflow Section
        self.assertIn(b"Academic Project Disclaimer", res.data)  # Footer disclaimer
        
        # Form field presence
        self.assertIn(b'name="person_age"', res.data)
        self.assertIn(b'name="person_income"', res.data)
        self.assertIn(b'name="person_home_ownership"', res.data)
        self.assertIn(b'name="person_emp_length"', res.data)
        self.assertIn(b'name="loan_intent"', res.data)
        self.assertIn(b'name="loan_amnt"', res.data)
        self.assertIn(b'name="loan_percent_income"', res.data)
        self.assertIn(b'name="cb_person_default_on_file"', res.data)
        self.assertIn(b'name="cb_person_cred_hist_length"', res.data)
        
    def test_no_leak_of_internal_secrets_and_paths(self):
        """Verify API response error bodies never expose local paths, stack traces, or keys."""
        # Cause validation failure
        res = self.client.post("/api/predict", json={"invalid_field": "val"})
        self.assertEqual(res.status_code, 400)
        
        self.assertNotIn("C:\\", res.text)
        self.assertNotIn("Traceback", res.text)
        self.assertNotIn("SECRET_KEY", res.text)
        
        # Test 404 page
        res = self.client.get("/unknown-browser-page")
        self.assertEqual(res.status_code, 404)
        self.assertNotIn("C:\\", res.text)

    def test_exact_feature_schema_and_ordering(self):
        """Verify the exact 9 features exist in predictor and in the correct order."""
        expected_schema = [
            "person_age",
            "person_income",
            "person_home_ownership",
            "person_emp_length",
            "loan_intent",
            "loan_amnt",
            "loan_percent_income",
            "cb_person_default_on_file",
            "cb_person_cred_hist_length"
        ]
        self.assertEqual(flask_app.predictor_instance.feature_names, expected_schema)

    def test_ratio_mismatch_api_rejection(self):
        """Verify that API requests with mismatching loan percentage of income are rejected."""
        mismatched_payload = self.low_risk_payload.copy()
        mismatched_payload["loan_percent_income"] = 0.50  # conflicts with 10000 / 120000 = 0.0833
        
        res = self.client.post("/api/predict", json=mismatched_payload)
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertFalse(data["success"])
        self.assertIn("does not align with computed ratio", data["error"]["message"])

    def test_risk_boundary_logic_and_threshold_distinction(self):
        """Verify boundary margins and binary threshold classification logic via mock predictions."""
        import numpy as np
        original_predict_proba = flask_app.predictor_instance.model.predict_proba
        try:
            # 1. Margins slightly below 0.12 (Low Risk)
            flask_app.predictor_instance.model.predict_proba = lambda x: np.array([[0.8801, 0.1199]])
            res = flask_app.predictor_instance.predict(self.low_risk_payload)
            self.assertEqual(res["risk_category"], "Low")
            self.assertEqual(res["predicted_class"], 0)
            
            # 2. Exactly 0.12 (Medium Risk)
            flask_app.predictor_instance.model.predict_proba = lambda x: np.array([[0.8800, 0.1200]])
            res = flask_app.predictor_instance.predict(self.low_risk_payload)
            self.assertEqual(res["risk_category"], "Medium")
            self.assertEqual(res["predicted_class"], 0)
            
            # 3. Slightly below 0.35 (Medium Risk)
            flask_app.predictor_instance.model.predict_proba = lambda x: np.array([[0.6501, 0.3499]])
            res = flask_app.predictor_instance.predict(self.low_risk_payload)
            self.assertEqual(res["risk_category"], "Medium")
            self.assertEqual(res["predicted_class"], 0)
            
            # 4. Exactly 0.35 (High Risk)
            flask_app.predictor_instance.model.predict_proba = lambda x: np.array([[0.6500, 0.3500]])
            res = flask_app.predictor_instance.predict(self.low_risk_payload)
            self.assertEqual(res["risk_category"], "High")
            self.assertEqual(res["predicted_class"], 0)
            
            # 5. Exactly 0.40 (High Risk Category but Class 0 Binary Verdict because threshold is 0.50)
            flask_app.predictor_instance.model.predict_proba = lambda x: np.array([[0.6000, 0.4000]])
            res = flask_app.predictor_instance.predict(self.low_risk_payload)
            self.assertEqual(res["risk_category"], "High")
            self.assertEqual(res["predicted_class"], 0)
            self.assertEqual(res["predicted_label"], "Lower Risk")
            
            # 6. Exactly 0.50 (Class 1 Binary Verdict)
            flask_app.predictor_instance.model.predict_proba = lambda x: np.array([[0.5000, 0.5000]])
            res = flask_app.predictor_instance.predict(self.low_risk_payload)
            self.assertEqual(res["predicted_class"], 1)
            self.assertEqual(res["predicted_label"], "Higher Risk")
        finally:
            flask_app.predictor_instance.model.predict_proba = original_predict_proba

    def test_shap_contribution_properties(self):
        """Verify SHAP factor signs and magnitude sorting order."""
        res = flask_app.explainer_instance.explain(self.low_risk_payload)
        self.assertTrue(res["success"])
        
        # Check increasing signs
        increasing = res["top_risk_increasing_factors"]
        for factor in increasing:
            self.assertGreater(factor["shap_value"], 0)
            self.assertEqual(factor["direction"], "increases_risk")
            
        # Check decreasing signs
        decreasing = res["top_risk_decreasing_factors"]
        for factor in decreasing:
            self.assertLess(factor["shap_value"], 0)
            self.assertEqual(factor["direction"], "decreases_risk")
            
        # Verify that increasing arrays are sorted by magnitude (descending)
        inc_vals = [abs(x["shap_value"]) for x in increasing]
        self.assertEqual(inc_vals, sorted(inc_vals, reverse=True))
        
        # Verify that decreasing arrays are sorted by magnitude (descending)
        dec_vals = [abs(x["shap_value"]) for x in decreasing]
        self.assertEqual(dec_vals, sorted(dec_vals, reverse=True))

    def test_startup_missing_files(self):
        """Verify CreditPredictor constructor error bounds on missing assets."""
        from src.prediction import CreditPredictor
        model_path = self.app.config["MODEL_PATH"]
        metadata_path = self.app.config["METADATA_PATH"]
        
        # Non-existent model path
        with self.assertRaises(FileNotFoundError):
            CreditPredictor(model_path="non_existent_file.joblib", metadata_path=metadata_path)
            
        # Non-existent metadata path
        with self.assertRaises(FileNotFoundError):
            CreditPredictor(model_path=model_path, metadata_path="non_existent_file.json")

if __name__ == "__main__":
    unittest.main()
