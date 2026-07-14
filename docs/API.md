# CrediSense AI REST API Documentation

This document describes the endpoints, input schemas, response structures, and HTTP status codes for the CrediSense AI credit risk REST API.

---

## 1. Input Data Schema

All API requests accepting payload bodies (`POST` requests) expect JSON formatted data with the following fields:

| Field Name | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `person_age` | Integer | `[18, 100]` | Applicant age. Negatives or values outside range are rejected. |
| `person_income` | Float/Int | `[0, inf)` | Annual income of the applicant. Negative values rejected. |
| `person_home_ownership` | String | `['RENT', 'OWN', 'MORTGAGE', 'OTHER']` | Case-insensitive home ownership status. |
| `person_emp_length` | Float/Int | `[0, inf)` | Duration of employment. Must satisfy `person_emp_length <= person_age - 14`. |
| `loan_intent` | String | `['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']` | Case-insensitive loan purpose. |
| `loan_amnt` | Float/Int | `(0, inf)` | Requested loan amount. Must be greater than zero. |
| `loan_percent_income` | Float | `[0.0, 1.0]` | Loan-to-income ratio (as a fraction, e.g., 0.2000). |
| `cb_person_default_on_file` | String | `['Y', 'N']` | Case-insensitive past credit default history flag. |
| `cb_person_cred_hist_length` | Integer | `[0, inf)` | Credit history duration. Negative values rejected. |

*Note: All fields are required. Excess fields in payloads are ignored. Submitting NaN or Infinity values triggers a validation error.*

---

## 2. API Endpoint Registry

### A. Health Monitoring Check
*   **Route**: `GET /health`
*   **Method**: `GET`
*   **HTTP Status Codes**:
    - `200`: Health status active (Predictor and/or Explainer running).
    - `503`: Prediction service unhealthy (Predictor model file failed loading).
*   **Example Response**:
    ```json
    {
        "status": "healthy",
        "prediction_service_available": true,
        "explanation_service_available": true,
        "metadata_available": true,
        "application_version": "1.0.0"
    }
    ```

---

### B. Retrieve Model Info
*   **Route**: `GET /api/model-info`
*   **Method**: `GET`
*   **HTTP Status Codes**:
    - `200`: Success.
    - `503`: Prediction service unavailable.
*   **Example Response**:
    ```json
    {
        "success": true,
        "model_info": {
            "project_name": "CrediSense AI: Explainable Credit Approval & Risk Assessment System",
            "model_type": "CalibratedClassifierCV(XGBClassifier)",
            "calibration_method": "isotonic",
            "number_of_features": 9,
            "feature_names": ["person_age", "person_income", "person_home_ownership", "person_emp_length", "loan_intent", "loan_amnt", "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"],
            "classification_threshold": 0.5,
            "risk_category_boundaries": {
                "low_risk_upper_bound": 0.12,
                "high_risk_lower_bound": 0.35
            },
            "test_metrics": {
                "Accuracy": 0.8928,
                "Precision": 0.9076,
                "Recall": 0.5677,
                "F1_Score": 0.6985,
                "ROC_AUC": 0.9094,
                "PR_AUC": 0.8263,
                "Brier_Score": 0.0806,
                "Log_Loss": 0.2736
            },
            "SHAP_explanation_scope": "underlying_xgboost_model_ensemble",
            "library_versions": {
                "xgboost": "3.2.0",
                "scikit-learn": "1.7.2",
                "joblib": "1.5.3"
            }
        }
    }
    ```

---

### C. Run Calibrated Risk Prediction
*   **Route**: `POST /api/predict`
*   **Method**: `POST`
*   **Headers**: `Content-Type: application/json`
*   **HTTP Status Codes**:
    - `200`: Success.
    - `400`: Input validation failure or bad format.
    - `503`: Service unavailable.
*   **Example Request Body**:
    ```json
    {
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
    ```
*   **Example Response**:
    ```json
    {
        "success": true,
        "prediction": {
            "risk_probability": 0.1254,
            "risk_percentage": 12.54,
            "predicted_class": 0,
            "predicted_label": "Lower Risk",
            "risk_category": "Medium",
            "classification_threshold": 0.5,
            "risk_boundaries": {
                "low_risk_upper_bound": 0.12,
                "high_risk_lower_bound": 0.35
            },
            "model_name": "CalibratedClassifierCV(XGBClassifier)",
            "disclaimer": "Disclaimer: CrediSense AI is an academic decision-support system designed for training and educational demonstration. It is not an automated banking decision maker and does not constitute a financial commitment or binding approval."
        }
    }
    ```

---

### D. Generate SHAP Explanation
*   **Route**: `POST /api/explain`
*   **Method**: `POST`
*   **Headers**: `Content-Type: application/json`
*   **HTTP Status Codes**:
    - `200`: Success (includes availability flag, handles graceful degradation).
    - `400`: Input validation failure.
    - `503`: Service unavailable.
*   **Example Response (SHAP Active)**:
    ```json
    {
        "success": true,
        "explanation_available": true,
        "explanation": {
            "explanation_scope": "underlying_xgboost_model_ensemble",
            "base_value": 0.0253,
            "top_risk_increasing_factors": [
                {
                    "technical_feature": "person_home_ownership_RENT",
                    "friendly_feature": "Home Ownership Status",
                    "value": "RENT",
                    "shap_value": 0.144,
                    "direction": "increases_risk",
                    "description": "Your home ownership status of 'RENT' increased the model's estimated risk."
                }
            ],
            "top_risk_decreasing_factors": [
                {
                    "technical_feature": "person_income",
                    "friendly_feature": "Annual Income",
                    "value": 60000.0,
                    "shap_value": -0.6083,
                    "direction": "decreases_risk",
                    "description": "Your reported annual income of $60,000.00 reduced the model's estimated risk."
                }
            ],
            "disclaimer": "Disclaimer: SHAP values explain feature contributions in the underlying XGBoost decision model..."
        }
    }
    ```
*   **Example Response (SHAP Disabled / Degraded)**:
    ```json
    {
        "success": true,
        "explanation_available": false,
        "message": "SHAP explanations are currently unavailable: [Inner Exception Error Message]",
        "prediction_summary": {
            "risk_probability": 0.1254,
            "risk_category": "Medium"
        }
    }
    ```

---

### E. Run Prediction and Explanation Combined
*   **Route**: `POST /api/predict-with-explanation`
*   **Method**: `POST`
*   **Headers**: `Content-Type: application/json`
*   **HTTP Status Codes**:
    - `200`: Success.
    - `400`: Input validation failure.
    - `503`: Service unavailable.
*   **Example Response**:
    ```json
    {
        "success": true,
        "prediction": {
            "risk_probability": 0.1254,
            "risk_percentage": 12.54,
            "predicted_class": 0,
            "predicted_label": "Lower Risk",
            "risk_category": "Medium",
            "classification_threshold": 0.5,
            "risk_boundaries": {
                "low_risk_upper_bound": 0.12,
                "high_risk_lower_bound": 0.35
            },
            "model_name": "CalibratedClassifierCV(XGBClassifier)",
            "disclaimer": "..."
        },
        "explanation_available": true,
        "explanation": {
            "explanation_scope": "underlying_xgboost_model_ensemble",
            "base_value": 0.0253,
            "top_risk_increasing_factors": [...],
            "top_risk_decreasing_factors": [...],
            "disclaimer": "..."
        },
        "explanation_message": null
    }
    ```

---

## 3. Structured Error Response Mappings

If requests fail, the API returns a structured error mapping with an appropriate HTTP status code.

### A. Schema Validation Failure (HTTP 400)
```json
{
    "success": false,
    "error": {
        "type": "validation_error",
        "message": "Employment length (15.0 years) is logically impossible for age 20 (violates age boundary)"
    }
}
```

### B. Endpoint Not Found (HTTP 404)
```json
{
    "success": false,
    "error": {
        "type": "not_found",
        "message": "The requested resource could not be found."
    }
}
```

### C. HTTP Method Not Allowed (HTTP 405)
```json
{
    "success": false,
    "error": {
        "type": "method_not_allowed",
        "message": "The HTTP method is not allowed for this endpoint."
    }
}
```

### D. Payload Entity Too Large (HTTP 413)
```json
{
    "success": false,
    "error": {
        "type": "payload_too_large",
        "message": "The uploaded payload size exceeds the server's limit."
    }
}
```

### E. Service Unavailable (Model Missing) (HTTP 503)
```json
{
    "success": false,
    "error": {
        "type": "service_unavailable",
        "message": "The prediction service is currently unavailable."
    }
}
```
