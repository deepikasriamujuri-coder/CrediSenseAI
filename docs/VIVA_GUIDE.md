# CrediSense AI: Viva Preparation Guide

This document contains answers to key questions commonly asked during academic project evaluations and viva examinations for final-year B.Tech projects.

---

### 1. What problem does CrediSense AI solve?
It addresses the opacity and unreliability of machine learning predictions in retail credit scoring. Standard models often produce raw scores that don't match actual default rates, and they lack transparency. CrediSense AI calibrates predicted default probabilities to match real-world rates and provides local explanations for every applicant.

---

### 2. Why is this a classification problem?
Because the goal is to predict which of two mutually exclusive classes an applicant belongs to: Class 0 (lower credit/default risk) or Class 1 (higher credit/default risk) based on historical observation.

---

### 3. What dataset did you use?
We used the traceable, CC0-licensed **Kaggle Credit Risk Dataset**. It contains 32,581 raw records representing standard variables collected at the time of a loan application.

---

### 4. What preprocessing was performed?
We built an isolated, pipeline-scoped preprocessor:
- **Numerical Columns**: Median imputation for missing values, followed by `RobustScaler` (scaling by median and IQR to handle outliers robustly).
- **Categorical Columns**: Mode imputation (most frequent value) followed by `OneHotEncoder` to encode categories as binary features.

---

### 5. Why XGBoost?
XGBoost (eXtreme Gradient Boosting) is a highly efficient implementation of gradient boosted decision trees. It excels at handling tabular data, handles non-linear relationships and missing values natively, and outperformed linear estimators significantly during cross-validation.

---

### 6. What models were compared?
During Phase 5 and 6, we compared:
1.  **Baseline Dummy Classifier** (Prior frequency benchmark)
2.  **Logistic Regression** (Linear baseline)
3.  **Decision Tree** (Non-linear single tree baseline)
4.  **Random Forest** (Bagged trees)
5.  **XGBoost (Default & Weighted)** (Boosting tree ensembles)

*Tuned XGBoost Weighted* was selected as the champion candidate based on training-set cross-validation PR-AUC (0.8109) and Recall (73.47%).

---

### 7. What is probability calibration?
Classification models like XGBoost output scores optimized for ranking rather than actual probability matching. Probability calibration maps these raw outputs so that the predicted probability matches the empirical frequency of default (e.g., of applicants predicted with a 20% risk, approximately 20% should actually default).

---

### 8. Why isotonic calibration?
We compared Platt Scaling (sigmoid fit) and Isotonic Regression (non-parametric step function fit). Isotonic Calibration achieved the lowest Brier Score (0.0849) on our validation slice, reducing probability errors by ~20.9% compared to the uncalibrated baseline.

---

### 9. What is the difference between classification threshold and risk boundaries?
- **Binary Classification Threshold (0.50)**: Used to split predictions into Class 0 (Lower Risk) and Class 1 (Higher Risk).
- **Policy Risk Boundaries (0.12 and 0.35)**: Selected based on out-of-fold empirical training data to support business decisions. Low Risk (under 12%) is suitable for automated approvals support, Medium Risk (12% to 35%) requires manual underwriter review, and High Risk (above 35%) has a high concentration of defaults (~79%).
- An applicant with a probability of 0.40 is classified as High Risk (business category) but still Class 0 (verdict), because 0.40 is below the 0.50 binary classification threshold.

---

### 10. What is SHAP?
SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain individual machine learning predictions. It assigns each feature a Shapley value representing its additive contribution to shift the prediction from the average baseline model output.

---

### 11. How are SHAP explanations generated?
We run `shap.TreeExplainer` on the underlying XGBoost decision trees. Since our model is a 5-fold ensemble (`CalibratedClassifierCV`), we average local SHAP values across all 5 estimators to produce a bagged explanation that matches the production model's architecture.

---

### 12. Why does SHAP not directly explain the calibrated probability?
SHAP values are calculated in the log-odds (margin) space of the raw XGBoost models. Isotonic calibration is a non-linear monotonic mapping applied downstream. While the SHAP values sum up exactly to the raw margin output, they do not add up directly to the calibrated probability due to this non-linear scaling.

---

### 13. What are risk-increasing factors?
Features that have positive SHAP values (log-odds > 0), shifting the model's prediction higher than the base expected value. Common examples include renting history, previous defaults, or a high loan-to-income ratio.

---

### 14. What are risk-decreasing factors?
Features that have negative SHAP values (log-odds < 0), shifting the prediction lower than the base expected value. Common examples include high annual income, low requested loan amounts, or stable home ownership.

---

### 15. How do you prevent invalid inputs?
We implement a two-layered validation:
1.  **Client-side (HTML5/JavaScript)**: Form inputs enforce minimum and maximum values (e.g. age between 18 and 100).
2.  **Server-side (Python validator)**: Authoritative boundary checks inside `CreditPredictor.validate_input` assert correct types, reject NaNs and infinities, check vocabulary categories, and enforce logical checks.

---

### 16. Why verify loan-to-income ratio server-side?
To prevent manual data tampering or rounding inconsistencies. The server computes the ratio `loan_amnt / person_income` directly. If the submitted ratio differs by more than 0.01, it is rejected; otherwise, it is normalized to the exact computed ratio.

---

### 17. What happens if SHAP fails?
The Flask application degrades gracefully. The global variable `SHAP_AVAILABLE` is toggled to `False`. Predictions and health endpoints remain functional, but explanation components display a safe warning card instead of crashing.

---

### 18. Why does the app still work without SHAP?
Because prediction and explanation are separated. The core model inference (`predict_proba`) does not depend on SHAP. SHAP is an optional post-hoc explanation layer.

---

### 19. What APIs exist?
- `GET /health` (System health checks)
- `GET /api/model-info` (Model parameters and test performance metadata)
- `POST /api/predict` (Calibrated probability predictions)
- `POST /api/explain` (SHAP explanation log-odds factors)
- `POST /api/predict-with-explanation` (Combined prediction and explanation pass)

---

### 20. How did you test the project?
We wrote automated tests using Python's `unittest` library. The suite covers:
- Data preprocessors
- Model parameter configs
- Predictor inputs and boundary validation checks
- SHAP sign logic and sorting orders
- API routes and status codes
- Startup failure file exceptions

---

### 21. What are the limitations?
1.  **Tabular data limit**: The model cannot process unstructured inputs like text notes or PDF bank statements.
2.  **Generalization limits**: The dataset is historical and does not reflect current inflationary adjustments.
3.  **SHAP scope**: SHAP details model feature weights, not real-world financial causality.

---

### 22. Is the system suitable for real bank lending decisions?
No. It is an academic prototype built for decision support. Live banking systems require strict regulatory audits, comprehensive fairness checks (e.g. Equal Credit Opportunity Act compliance), and continuous model drift monitoring.

---

### 23. What improvements would you make?
- Adding external validation datasets to test domain shift.
- Implementing a comprehensive fairness audit tool to monitor disparate impact.
- Adding drift monitors to flag changes in default rate ratios over time.

---

### 24. Explain the complete architecture.
The input is received via a browser form or REST JSON. It is validated and preprocessed inside a pipeline. The tuned XGBoost model outputs raw scores, which are scaled to calibrated probabilities using Isotonic calibration. The underlying trees generate log-odds SHAP values, which are averaged across ensemble splits, mapped to friendly feature names, and rendered on the Flask dashboard.

---

### 25. Explain one prediction from input to result.
An applicant submits: Annual Income = $120k, Loan Amount = $10k.
1.  The validator confirms variables are positive and checks the loan-to-income ratio (0.0833).
2.  The preprocessor imputes categories and robust-scales numerical features.
3.  The calibrated classifier ensemble runs the transformed array through its 5 underlying folds.
4.  The average calibrated default probability is scored (0.47%), mapping it to Low Risk (< 12%) and Class 0 (Lower Risk).
5.  SHAP calculates that the high income and low loan-to-income ratio significantly reduced default risk, outputting negative log-odds weights.
6.  The result dashboard displays the calibrated score, positioning it on the risk gauge, and lists the top SHAP drivers.
