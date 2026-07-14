import unittest
import os
import json
import joblib
import pandas as pd
import numpy as np
from src.prediction import CreditPredictor
from src.preprocessing import TARGET_COLUMN

class TestProductionPredictionPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.model_path = "models/credit_risk_model.joblib"
        cls.metadata_path = "models/model_metadata.json"
        
        if not os.path.exists(cls.model_path) or not os.path.exists(cls.metadata_path):
            raise FileNotFoundError("Model and metadata must be persisted before running tests.")
            
        cls.predictor = CreditPredictor(model_path=cls.model_path, metadata_path=cls.metadata_path)
        
        # Reference valid input
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
        
    def test_model_and_metadata_loading(self):
        """Verify model and metadata artifacts deserialization."""
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.metadata)
        self.assertEqual(self.predictor.metadata["model_type"], "CalibratedClassifierCV(XGBClassifier)")
        
    def test_valid_prediction(self):
        """Verify that a standard valid input generates a successful prediction output."""
        res = self.predictor.predict(self.valid_input)
        self.assertTrue(res["success"])
        self.assertIn("risk_probability", res)
        self.assertIn("risk_category", res)
        self.assertIn("predicted_class", res)
        self.assertIn("predicted_label", res)
        self.assertIn("disclaimer", res)
        
    def test_calibrated_probability_range(self):
        """Verify probability outputs fall strictly between [0.0, 1.0]."""
        res = self.predictor.predict(self.valid_input)
        prob = res["risk_probability"]
        self.assertTrue(0.0 <= prob <= 1.0)
        
    def test_binary_classification_logic(self):
        """Verify that classification labels align with classification thresholds."""
        res = self.predictor.predict(self.valid_input)
        prob = res["risk_probability"]
        pred_class = res["predicted_class"]
        pred_label = res["predicted_label"]
        thresh = res["classification_threshold"]
        
        if prob >= thresh:
            self.assertEqual(pred_class, 1)
            self.assertEqual(pred_label, "Higher Risk")
        else:
            self.assertEqual(pred_class, 0)
            self.assertEqual(pred_label, "Lower Risk")
            
    def test_risk_category_assignment(self):
        """Verify risk categories correspond to metadata boundaries."""
        res = self.predictor.predict(self.valid_input)
        prob = res["risk_probability"]
        cat = res["risk_category"]
        
        low_bound = self.predictor.low_risk_upper
        high_bound = self.predictor.high_risk_lower
        
        if prob < low_bound:
            self.assertEqual(cat, "Low")
        elif prob >= high_bound:
            self.assertEqual(cat, "High")
        else:
            self.assertEqual(cat, "Medium")
            
    def test_missing_field_rejection(self):
        """Verify predictor rejects inputs missing essential features."""
        for field in self.predictor.feature_names:
            bad_input = self.valid_input.copy()
            bad_input.pop(field)
            res = self.predictor.predict(bad_input)
            self.assertFalse(res["success"])
            self.assertIn("Missing required field", res["error"])
            
    def test_invalid_numerical_value_rejection(self):
        """Verify predictor rejects negative numbers, illegal bounds, and bad formats."""
        # Negative Age
        bad_input = self.valid_input.copy()
        bad_input["person_age"] = -5
        res = self.predictor.predict(bad_input)
        self.assertFalse(res["success"])
        
        # Age out of range
        bad_input = self.valid_input.copy()
        bad_input["person_age"] = 14
        res = self.predictor.predict(bad_input)
        self.assertFalse(res["success"])
        
        # Negative Income
        bad_input = self.valid_input.copy()
        bad_input["person_income"] = -1000
        res = self.predictor.predict(bad_input)
        self.assertFalse(res["success"])
        
        # Impossible employment length vs age
        bad_input = self.valid_input.copy()
        bad_input["person_age"] = 20
        bad_input["person_emp_length"] = 15.0  # started work at age 5
        res = self.predictor.predict(bad_input)
        self.assertFalse(res["success"])
        self.assertIn("employment length", res["error"].lower())
        
        # String instead of numeric
        bad_input = self.valid_input.copy()
        bad_input["loan_amnt"] = "five-thousand"
        res = self.predictor.predict(bad_input)
        self.assertFalse(res["success"])
        
    def test_nan_rejection(self):
        """Verify predictor explicitly rejects NaN values."""
        bad_input = self.valid_input.copy()
        bad_input["person_income"] = float("nan")
        res = self.predictor.predict(bad_input)
        self.assertFalse(res["success"])
        self.assertIn("NaN and Infinity are not allowed", res["error"])
        
    def test_infinity_rejection(self):
        """Verify predictor explicitly rejects Infinity values."""
        bad_input = self.valid_input.copy()
        bad_input["loan_amnt"] = float("inf")
        res = self.predictor.predict(bad_input)
        self.assertFalse(res["success"])
        self.assertIn("NaN and Infinity are not allowed", res["error"])
        
    def test_invalid_categorical_rejection(self):
        """Verify predictor rejects category values outside vocabulary list."""
        bad_input = self.valid_input.copy()
        bad_input["person_home_ownership"] = "JUNK_CATEGORY"
        res = self.predictor.predict(bad_input)
        self.assertFalse(res["success"])
        self.assertIn("person_home_ownership", res["error"])
        
    def test_production_feature_exclusions(self):
        """Verify post-assessment columns (loan_grade, loan_int_rate) are not in predictor features list."""
        self.assertNotIn("loan_grade", self.predictor.feature_names)
        self.assertNotIn("loan_int_rate", self.predictor.feature_names)
        
    def test_feature_ordering_preservation(self):
        """Verify that pandas DataFrame mapping preserves exact training schema ordering."""
        # Convert valid dict using predictor columns
        df_input = pd.DataFrame([self.valid_input], columns=self.predictor.feature_names)
        self.assertEqual(list(df_input.columns), self.predictor.feature_names)
        
    def test_ratio_consistency_verification(self):
        """Verify that conflicting loan-to-income ratios are rejected and valid ones are cleaned."""
        # Significant mismatch: Income 60,000, Loan 10,000, submitted ratio 0.50 (instead of ~0.1667)
        bad_input = self.valid_input.copy()
        bad_input["loan_percent_income"] = 0.50
        res = self.predictor.predict(bad_input)
        self.assertFalse(res["success"])
        self.assertIn("does not align with computed ratio", res["error"])
        
        # Valid close match: submitted ratio 0.1660 (which is within 0.01 of 10000/60000 = 0.1667)
        ok_input = self.valid_input.copy()
        ok_input["loan_percent_income"] = 0.1660
        res = self.predictor.predict(ok_input)
        self.assertTrue(res["success"])
        
        # Verify validate_input directly cleans and aligns ratio to computed ratio
        is_valid, err, cleaned = self.predictor.validate_input(ok_input)
        self.assertTrue(is_valid)
        self.assertEqual(cleaned["loan_percent_income"], 0.1667)
        
        # Zero income and positive loan
        bad_input_zero = self.valid_input.copy()
        bad_input_zero["person_income"] = 0
        res = self.predictor.predict(bad_input_zero)
        self.assertFalse(res["success"])
        self.assertIn("Income is 0, but requested loan amount is positive", res["error"])

if __name__ == "__main__":
    unittest.main()
