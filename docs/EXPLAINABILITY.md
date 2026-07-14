# Explainable AI (XAI) Architecture: SHAP Integration

This document outlines the technical design, integration strategy, and responsible interpretation guidelines for explainability in the CrediSense AI credit risk decision-support system.

---

## 1. Distinction: Probability Calibration vs. SHAP Explanation

To prevent confusion when presenting model outputs to final users:
*   **Calibrated Risk Probability**: Used for the user-facing probability score and Low/Medium/High risk category. It represents the statistically aligned, empirical probability of default based on **Isotonic Calibration** (fitted via `CalibratedClassifierCV` on training data).
*   **SHAP Explanation**: Explains feature contributions in the underlying **XGBoost decision model**. SHAP values are computed in the raw log-odds (margin) space of the boosting trees. They represent how features push the decision model's risk score higher or lower relative to the model's average base value.
*   **Additivity**: The SHAP values sum up exactly to the raw margin output of the underlying XGBoost model (`base_value + sum(shap_values) = raw_margin`). They do not sum up directly to the final calibrated probability since Isotonic Calibration is a non-linear monotonic transformation applied downstream.

---

## 2. Multi-Estimator Explanation Strategy

Since our production model is a `CalibratedClassifierCV` ensemble consisting of **5 estimators** trained via cross-validation, explaining a single tree model would create a discrepancy.
To address this:
1.  We extract the 5 fitted internal pipelines from the calibrated classifier ensemble (`calibrated_classifiers_[i].estimator`).
2.  For a given raw applicant input, we preprocess the input 5 times using the respective fitted preprocessing transformer of each fold.
3.  We run `shap.TreeExplainer` on each of the 5 underlying `XGBClassifier` models.
4.  We average the resulting SHAP values and base values across the 5 estimators to produce a **bagged SHAP explanation** that matches the ensemble architecture of the production model. This yields highly stable, robust local explanations.

---

## 3. Transformed Feature Name Recovery & Mapping

Features entering the model are preprocessed:
- **Numerical Features** are scaled using `RobustScaler`.
- **Categorical Features** are encoded using `OneHotEncoder`.

To recover original columns:
1.  We extract feature names from the fitted `ColumnTransformer` (which yields names like `person_income` or `person_home_ownership_RENT`).
2.  Numerical features are matched back to their raw values in the input data to display actual currencies and numbers in descriptions (e.g. `person_income` -> "Annual Income", showing `$45,000.00`).
3.  Categorical one-hot encoded features (e.g. `person_home_ownership_RENT`) are checked for active values (`1.0`). If active, they are mapped to human-readable strings (e.g., "Home Ownership Status: RENT"). Inactive categorical variables are filtered out to keep explanations concise.
4.  Technical names (like `cat__loan_intent_MEDICAL` or `num__person_income`) are never exposed to the end-user.

---

## 4. Global Explainability Methodology

We evaluated global model behavior on a representative training sample of **500 records** (selected using `random_state=42` to ensure reproducibility):
*   **Summary Beeswarm Plot**: Plots features ordered by their overall impact, showing how high/low feature values drive positive/negative log-odds risk shifts.
*   **Bar Importance Plot**: Summarizes features based on their mean absolute SHAP value, identifying which variables have the largest average impact.
*   *Plots Saved under*: `static/images/shap/`

### Built-in Gain vs. SHAP Importance
A comparison of built-in XGBoost gain importance against mean absolute SHAP values is saved in `models/evaluation/feature_importance_comparison.csv`.
*   **XGBoost Built-in Gain**: Measures how much splits on a feature reduce the training loss function *locally* during tree construction. It is highly dependent on split sequence and can underestimate the impact of variables that are split fewer times but carry large values.
*   **Mean Absolute SHAP**: Measures the average actual additive effect of features on the prediction scale across the representative sample, yielding a more consistent global feature ranking.

---

## 5. Local Explanation Structure

The `CreditRiskExplainer` returns a structured dictionary:
```json
{
    "success": true,
    "explanation_scope": "underlying_xgboost_model_ensemble",
    "calibrated_risk_probability": 0.0773,
    "risk_category": "Low",
    "base_value": 0.0253,
    "additivity_verified": true,
    "top_risk_increasing_factors": [
        {
            "technical_feature": "loan_intent_MEDICAL",
            "friendly_feature": "Loan Purpose",
            "value": "MEDICAL",
            "shap_value": 0.1478,
            "direction": "increases_risk",
            "description": "Your declared loan purpose of 'MEDICAL' increased the model's estimated risk."
        }
    ],
    "top_risk_decreasing_factors": [
        {
            "technical_feature": "person_income",
            "friendly_feature": "Annual Income",
            "value": 45000.0,
            "shap_value": -0.6083,
            "direction": "decreases_risk",
            "description": "Your reported annual income of $45,000.00 reduced the model's estimated risk."
        }
    ],
    "disclaimer": "..."
}
```

---

## 6. Responsible Interpretation Guidance

Explainability output must be presented with scientific caution:
*   **Estimated Risk influence**: Use language such as *"increased the model's estimated risk"* or *"reduced the model's estimated risk"*.
*   **No Causal Proofs**: Do not claim that features *caused* a rejection, or *prove* default capability. SHAP values indicate statistical association within the decision boundary, not physical causality.
*   **No Personal Judgments**: Avoid phrases suggesting financial irresponsibility. Treat explanations strictly as academic decision-support values.

---

## 7. Compatibility Workaround & Pinned Library Versions

Modern XGBoost (`3.2.0`) serializes tree models using UBJSON formatting. In this format, parameters like `base_score` are written as string-serialized arrays (e.g. `"[5.0000924E-1]"`). When `shap.TreeExplainer` parses this via the python float compiler in `decode_ubjson_buffer`, it raises a `ValueError`.

To bypass this parser crash, we apply a private internal monkeypatch on startup:
```python
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
```

### Pinned Package Specifications
To prevent compatibility drift, the environment requirements are strictly pinned:
*   `xgboost==3.2.0`
*   `shap==0.49.1`
*   `scikit-learn==1.7.2`
*   `flask==3.1.3`
*   `pandas==2.3.3`
*   `numpy==2.2.6`

---

## 8. Technical Limitations of SHAP Scope

1.  **Model Decision Space Limits**: SHAP values explain log-odds shifts in the underlying raw trees before sigmoid activation or isotonic scaling are applied. They do not map linearly to calibrated probability adjustments.
2.  **No Causal Guidance**: SHAP details feature weights inside the learned splits of our model. It does not reflect a physical or legal cause-and-effect relationship in real-world lending parameters.
3.  **Local Approximation Drift**: Extremely unusual feature combinations can produce local SHAP explanations that reflect model extrapolation rather than actual predictive characteristics.

