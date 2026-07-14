import os
import json
import joblib
import pandas as pd
import numpy as np

class CreditPredictor:
    def __init__(self, model_path="models/credit_risk_model.joblib", metadata_path="models/model_metadata.json"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
            
        self.model = joblib.load(model_path)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
            
        self.feature_names = [
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
        
        # Valid categories derived from training set
        self.valid_home_ownerships = {"RENT", "OWN", "MORTGAGE", "OTHER"}
        self.valid_intents = {"PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"}
        self.valid_defaults = {"Y", "N"}
        
        # Risk thresholds from metadata
        self.classification_threshold = self.metadata["classification_threshold"]
        self.low_risk_upper = self.metadata["risk_category_boundaries"]["low_risk_upper_bound"]
        self.high_risk_lower = self.metadata["risk_category_boundaries"]["high_risk_lower_bound"]
        
    def validate_input(self, data):
        """
        Validates the raw dictionary input.
        Returns (is_valid, error_message, cleaned_dict)
        """
        if not isinstance(data, dict):
            return False, "Input data must be a dictionary", None
            
        cleaned = {}
        
        # Check for missing fields
        for field in self.feature_names:
            if field not in data:
                return False, f"Missing required field: '{field}'", None
                
            val = data[field]
            
            # Reject NaN and Infinity
            try:
                if isinstance(val, (int, float)) and (np.isnan(val) or np.isinf(val)):
                    return False, f"Invalid value in '{field}': NaN and Infinity are not allowed", None
            except TypeError:
                pass
                
            cleaned[field] = val
            
        # Numerical Conversions & Range Checks
        
        # 1. person_age
        try:
            cleaned["person_age"] = int(cleaned["person_age"])
        except (ValueError, TypeError):
            return False, "Field 'person_age' must be a valid integer", None
            
        if cleaned["person_age"] < 18 or cleaned["person_age"] > 100:
            return False, "Age must be between 18 and 100", None
            
        # 2. person_income
        try:
            cleaned["person_income"] = float(cleaned["person_income"])
        except (ValueError, TypeError):
            return False, "Field 'person_income' must be a valid number", None
            
        if cleaned["person_income"] < 0:
            return False, "Income cannot be negative", None
            
        # 3. person_emp_length
        try:
            cleaned["person_emp_length"] = float(cleaned["person_emp_length"])
        except (ValueError, TypeError):
            return False, "Field 'person_emp_length' must be a valid number", None
            
        if cleaned["person_emp_length"] < 0:
            return False, "Employment length cannot be negative", None
            
        # Logical check: emp_length <= age - 14
        if cleaned["person_emp_length"] > (cleaned["person_age"] - 14):
            return False, f"Employment length ({cleaned['person_emp_length']} years) is logically impossible for age {cleaned['person_age']} (violates age boundary)", None
            
        # 4. loan_amnt
        try:
            cleaned["loan_amnt"] = float(cleaned["loan_amnt"])
        except (ValueError, TypeError):
            return False, "Field 'loan_amnt' must be a valid number", None
            
        if cleaned["loan_amnt"] <= 0:
            return False, "Loan amount must be greater than zero", None
            
        # 5. loan_percent_income
        try:
            cleaned["loan_percent_income"] = float(cleaned["loan_percent_income"])
        except (ValueError, TypeError):
            return False, "Field 'loan_percent_income' must be a valid number", None
            
        if cleaned["loan_percent_income"] < 0.0 or cleaned["loan_percent_income"] > 1.0:
            return False, "Loan percent of income must be a fraction between 0.0 and 1.0", None
            
        # Server-side consistency validation check
        if cleaned["person_income"] > 0:
            computed_ratio = cleaned["loan_amnt"] / cleaned["person_income"]
            if abs(cleaned["loan_percent_income"] - computed_ratio) > 0.01:
                return False, f"Submitted loan-to-income ratio ({cleaned['loan_percent_income']}) does not align with computed ratio ({computed_ratio:.4f}) based on income and loan amount", None
            # Override with exact recomputed value to avoid floating precision discrepancy
            cleaned["loan_percent_income"] = float(np.round(computed_ratio, 4))
        else:
            if cleaned["loan_amnt"] > 0:
                return False, "Income is 0, but requested loan amount is positive. Loan-to-income ratio is invalid.", None
            
        # 6. cb_person_cred_hist_length
        try:
            cleaned["cb_person_cred_hist_length"] = int(cleaned["cb_person_cred_hist_length"])
        except (ValueError, TypeError):
            return False, "Field 'cb_person_cred_hist_length' must be a valid integer", None
            
        if cleaned["cb_person_cred_hist_length"] < 0:
            return False, "Credit history length cannot be negative", None
            
        # Categorical String Checks
        
        # 7. person_home_ownership
        val_home = str(cleaned["person_home_ownership"]).upper().strip()
        if val_home not in self.valid_home_ownerships:
            return False, f"Invalid value in 'person_home_ownership'. Must be one of: {list(self.valid_home_ownerships)}", None
        cleaned["person_home_ownership"] = val_home
        
        # 8. loan_intent
        val_intent = str(cleaned["loan_intent"]).upper().strip()
        if val_intent not in self.valid_intents:
            return False, f"Invalid value in 'loan_intent'. Must be one of: {list(self.valid_intents)}", None
        cleaned["loan_intent"] = val_intent
        
        # 9. cb_person_default_on_file
        val_def = str(cleaned["cb_person_default_on_file"]).upper().strip()
        if val_def not in self.valid_defaults:
            return False, f"Invalid value in 'cb_person_default_on_file'. Must be one of: {list(self.valid_defaults)}", None
        cleaned["cb_person_default_on_file"] = val_def
        
        return True, "", cleaned

    def predict(self, raw_input):
        """
        Accepts a dictionary of raw input data, validates it, matches feature order,
        runs the production pipeline, and returns a structured risk outcome dictionary.
        """
        is_valid, err_msg, cleaned_data = self.validate_input(raw_input)
        if not is_valid:
            return {
                "success": False,
                "error": err_msg
            }
            
        # Convert dictionary to DataFrame with the exact feature ordering used during training
        df_input = pd.DataFrame([cleaned_data], columns=self.feature_names)
        
        # Predict probability of default (Class 1) using calibrated classifier
        try:
            prob = self.model.predict_proba(df_input)[0, 1]
        except Exception as e:
            return {
                "success": False,
                "error": f"Model inference execution error: {str(e)}"
            }
            
        # Determine classification class based on threshold
        pred_class = 1 if prob >= self.classification_threshold else 0
        pred_label = "Higher Risk" if pred_class == 1 else "Lower Risk"
        
        # Determine user-facing risk category
        if prob < self.low_risk_upper:
            risk_category = "Low"
        elif prob >= self.high_risk_lower:
            risk_category = "High"
        else:
            risk_category = "Medium"
            
        disclaimer = (
            "Disclaimer: CrediSense AI is an academic decision-support system designed "
            "for training and educational demonstration. It is not an automated banking "
            "decision maker and does not constitute a financial commitment or binding approval."
        )
        
        return {
            "success": True,
            "risk_probability": float(np.round(prob, 4)),
            "risk_percentage": float(np.round(prob * 100, 2)),
            "predicted_class": int(pred_class),
            "predicted_label": pred_label,
            "risk_category": risk_category,
            "classification_threshold": float(self.classification_threshold),
            "model_name": self.metadata["model_type"],
            "disclaimer": disclaimer
        }
