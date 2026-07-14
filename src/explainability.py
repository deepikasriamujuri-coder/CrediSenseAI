import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Monkeypatch SHAP to handle XGBoost 3.x base_score bracket format
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

# Import prediction validator to reuse rules
from src.prediction import CreditPredictor

class CreditRiskExplainer:
    def __init__(self, model_path="models/credit_risk_model.joblib", metadata_path="models/model_metadata.json"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
            
        self.predictor = CreditPredictor(model_path=model_path, metadata_path=metadata_path)
        self.model = self.predictor.model
        self.metadata = self.predictor.metadata
        
        # There are 5 estimators in the CalibratedClassifierCV ensemble
        self.estimators = [clf.estimator for clf in self.model.calibrated_classifiers_]
        
        # Initialize 5 TreeExplainers, one for each XGBoost model in the ensemble
        self.explainers = []
        for pipeline in self.estimators:
            xgb_model = pipeline.named_steps["classifier"]
            self.explainers.append(shap.TreeExplainer(xgb_model))
            
        self.feature_names = self.predictor.feature_names
        
        # Transformed feature names recovered from the first fitted preprocessor
        first_preprocessor = self.estimators[0].named_steps["preprocessor"]
        from src.preprocessing import get_feature_names
        self.transformed_features = get_feature_names(first_preprocessor)
        
        # Numerical features list for name matching
        self.numerical_features_set = {
            "person_age", "person_income", "person_emp_length", 
            "loan_amnt", "loan_percent_income", "cb_person_cred_hist_length"
        }
        
        # User-friendly explanation mappings
        self.friendly_names = {
            "person_age": "Age",
            "person_income": "Annual Income",
            "person_emp_length": "Employment Duration",
            "loan_amnt": "Loan Amount",
            "loan_percent_income": "Loan-to-Income Ratio",
            "cb_person_cred_hist_length": "Credit History Length",
            "person_home_ownership": "Home Ownership Status",
            "loan_intent": "Loan Purpose",
            "cb_person_default_on_file": "Previous Default Record"
        }
        
    def map_feature_to_friendly(self, technical_name):
        """
        Maps technical transformed feature names to human-readable names and values.
        """
        if technical_name in self.numerical_features_set:
            return self.friendly_names.get(technical_name, technical_name), ""
            
        # Categorical OHE handling
        for cat_field in ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]:
            if technical_name.startswith(cat_field + "_"):
                category_value = technical_name.replace(cat_field + "_", "")
                if category_value == "Y":
                    category_value = "Yes"
                elif category_value == "N":
                    category_value = "No"
                return self.friendly_names.get(cat_field, cat_field), category_value
        return technical_name, ""

    def get_friendly_description(self, technical_name, value, shap_value):
        """
        Generates user-friendly descriptive explanations based on feature direction.
        """
        friendly_name, friendly_val = self.map_feature_to_friendly(technical_name)
        direction = "increased the model's estimated risk" if shap_value > 0 else "reduced the model's estimated risk"
        
        if "person_age" in technical_name:
            return f"Your age ({int(value)} years) {direction}."
        elif "person_income" in technical_name:
            return f"Your reported annual income of ${value:,.2f} {direction}."
        elif "person_emp_length" in technical_name:
            return f"Your employment duration of {value:.1f} years {direction}."
        elif "loan_amnt" in technical_name:
            return f"The requested loan amount of ${value:,.2f} {direction}."
        elif "loan_percent_income" in technical_name:
            return f"The requested loan represents {value*100:.1f}% of your annual income, which {direction}."
        elif "cb_person_cred_hist_length" in technical_name:
            return f"Your credit history length of {int(value)} years {direction}."
        elif "person_home_ownership" in technical_name:
            return f"Your home ownership status of '{friendly_val}' {direction}."
        elif "loan_intent" in technical_name:
            return f"Your declared loan purpose of '{friendly_val}' {direction}."
        elif "cb_person_default_on_file" in technical_name:
            return f"A history of previous default = '{friendly_val}' {direction}."
            
        return f"Feature '{friendly_name}' ({value}) {direction}."

    def explain(self, raw_input, precomputed_prediction=None):
        """
        Accepts a dictionary of raw input, validates it, runs the explanation pipeline,
        and returns a structured dictionary containing SHAP details and descriptions.
        """
        is_valid, err_msg, cleaned_data = self.predictor.validate_input(raw_input)
        if not is_valid:
            return {
                "success": False,
                "error": err_msg
            }
            
        if precomputed_prediction is not None:
            pred_res = precomputed_prediction
        else:
            pred_res = self.predictor.predict(cleaned_data)
            
        df_input = pd.DataFrame([cleaned_data], columns=self.feature_names)
        
        all_shap_values = []
        all_base_values = []
        all_additivity_checks = []
        
        for i, pipeline in enumerate(self.estimators):
            preprocessor = pipeline.named_steps["preprocessor"]
            classifier = pipeline.named_steps["classifier"]
            explainer = self.explainers[i]
            
            X_trans = preprocessor.transform(df_input)
            shap_res = explainer(X_trans)
            
            shap_vals = shap_res.values[0]
            base_val = shap_res.base_values
            if hasattr(base_val, "__len__"):
                base_val = base_val[0]
                
            all_shap_values.append(shap_vals)
            all_base_values.append(base_val)
            
            raw_margin = classifier.predict(X_trans, output_margin=True)[0]
            summed_shap = base_val + np.sum(shap_vals)
            is_additive = np.isclose(raw_margin, summed_shap, atol=1e-4)
            all_additivity_checks.append(is_additive)
            
        avg_shap_values = np.mean(all_shap_values, axis=0)
        avg_base_value = float(np.mean(all_base_values))
        additivity_verified = all(all_additivity_checks)
        
        first_trans = self.estimators[0].named_steps["preprocessor"].transform(df_input)
        
        features_explanation = []
        for idx, feat_name in enumerate(self.transformed_features):
            shap_val = float(avg_shap_values[idx])
            trans_val = first_trans[0, idx]
            
            # Determine raw display value
            raw_display_val = trans_val
            if feat_name in self.numerical_features_set:
                raw_display_val = cleaned_data[feat_name]
            else:
                for cat_field in ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]:
                    if feat_name.startswith(cat_field + "_"):
                        category_value = feat_name.replace(cat_field + "_", "")
                        if trans_val == 1.0:
                            raw_display_val = category_value
                        else:
                            raw_display_val = "Not " + category_value
            
            friendly_name, friendly_val = self.map_feature_to_friendly(feat_name)
            desc = self.get_friendly_description(feat_name, raw_display_val, shap_val)
            
            # Filter inactive OHE columns to keep output tidy
            is_active = True
            if feat_name not in self.numerical_features_set and trans_val == 0.0:
                is_active = False
                
            features_explanation.append({
                "technical_feature": feat_name,
                "friendly_feature": friendly_name,
                "value": raw_display_val,
                "shap_value": float(np.round(shap_val, 4)),
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
                "description": desc,
                "absolute_magnitude": abs(shap_val),
                "is_active_category": is_active
            })
            
        active_explanations = [f for f in features_explanation if f["is_active_category"]]
        
        increasing = [f for f in active_explanations if f["shap_value"] > 0]
        decreasing = [f for f in active_explanations if f["shap_value"] < 0]
        
        increasing = sorted(increasing, key=lambda x: x["absolute_magnitude"], reverse=True)
        decreasing = sorted(decreasing, key=lambda x: x["absolute_magnitude"], reverse=True)
        
        for item in increasing + decreasing:
            item.pop("absolute_magnitude")
            item.pop("is_active_category")
            
        disclaimer = (
            "Disclaimer: SHAP values explain feature contributions in the underlying "
            "XGBoost decision model. The calibrated risk probability and risk category "
            "are generated using Isotonic Calibration, which improves probability reliability. "
            "SHAP explanations are not causal proofs and should be interpreted as decision support only."
        )
        
        return {
            "success": True,
            "explanation_scope": "underlying_xgboost_model_ensemble",
            "calibrated_risk_probability": pred_res["risk_probability"],
            "risk_category": pred_res["risk_category"],
            "base_value": float(np.round(avg_base_value, 4)),
            "additivity_verified": additivity_verified,
            "top_risk_increasing_factors": increasing[:4],
            "top_risk_decreasing_factors": decreasing[:4],
            "disclaimer": disclaimer
        }
