# CrediSense AI: Demo Guide & Video Script

This guide contains validated applicant profiles run directly against the production calibrated classifier and details a script plan for a 3-to-5-minute project presentation.

---

## 1. Validated Demonstration Applicant Profiles

Use these profiles during local demonstrations or video recording to display the model's calibration and threshold behavior:

### A. Low Risk Profile
*   **Input Profile**:
    - Age: `35` years
    - Annual Income: `$120,000`
    - Home Ownership: `MORTGAGE`
    - Employment Length: `10.0` years
    - Loan Purpose: `VENTURE`
    - Requested Amount: `$10,000`
    - Loan-to-Income Ratio: `0.0833`
    - Previous Defaults: `No (N)`
    - Credit History Length: `8` years
*   **Model Prediction Outputs**:
    - Calibrated Probability of Default: **0.47%**
    - Risk Category: **Low Risk**
    - Binary Verdict: **Lower Risk (Class 0)**
*   **Major SHAP Factor Drivers**:
    - *Risk Decreasing*: High annual income ($120k) and a low loan-to-income ratio (8.3%) are the strongest indicators pushing estimated risk down.
    - *Risk Increasing*: Minimal mortgage balance impact.

---

### B. Medium Risk Profile
*   **Input Profile**:
    - Age: `28` years
    - Annual Income: `$40,000`
    - Home Ownership: `RENT`
    - Employment Length: `2.0` years
    - Loan Purpose: `DEBTCONSOLIDATION`
    - Requested Amount: `$12,000`
    - Loan-to-Income Ratio: `0.3000`
    - Previous Defaults: `Yes (Y)`
    - Credit History Length: `4` years
*   **Model Prediction Outputs**:
    - Calibrated Probability of Default: **24.21%**
    - Risk Category: **Medium Risk** (Requires manual underwriter review)
    - Binary Verdict: **Lower Risk (Class 0)**
*   **Major SHAP Factor Drivers**:
    - *Risk Increasing*: Previous default history (Yes) and renting status push estimated risk higher.
    - *Risk Decreasing*: Medium employment length (2 years) slightly reduces risk.

---

### C. High Risk Profile
*   **Input Profile**:
    - Age: `22` years
    - Annual Income: `$20,000`
    - Home Ownership: `RENT`
    - Employment Length: `1.0` years
    - Loan Purpose: `DEBTCONSOLIDATION`
    - Requested Amount: `$15,000`
    - Loan-to-Income Ratio: `0.7500`
    - Previous Defaults: `Yes (Y)`
    - Credit History Length: `2` years
*   **Model Prediction Outputs**:
    - Calibrated Probability of Default: **100.00%**
    - Risk Category: **High Risk**
    - Binary Verdict: **Higher Risk (Class 1)**
*   **Major SHAP Factor Drivers**:
    - *Risk Increasing*: Extremely high loan-to-income ratio (75% of income requested) and previous default history.

---

### D. Form Validation Error Profile (Rejection Check)
*   **Input Profile**:
    - Age: `15` years (Age < 18)
    - Annual Income: `$60,000`
    - Home Ownership: `RENT`
    - Employment Length: `5.0` years
    - Loan Purpose: `PERSONAL`
    - Requested Amount: `$10,000`
    - Loan-to-Income Ratio: `0.1667`
    - Previous Defaults: `No (N)`
    - Credit History Length: `5` years
*   **Rejection Behavior**:
    - *Browser Outcome*: Renders a clean `error.html` page detailing that age must be between 18 and 100.
    - *REST API Outcome*: Returns HTTP Status `400` with:
      `{"success": false, "error": {"type": "validation_error", "message": "Age must be between 18 and 100"}}`

---

## 2. Project Demonstration Video Script (5-Minute Timeline)

### **0:00–0:20 | Project Introduction**
*   **Visual**: Screen showing the homepage top banner.
*   **Voice**: *"Hello. Welcome to the demonstration of CrediSense AI, an Explainable Credit Risk Assessment and Decision Support system built as a B.Tech final-year machine learning project. The system predicts default probability, maps it to policy-based risk categories, and explains predictions using SHAP."*

### **0:20–0:50 | Problem Statement & Objective**
*   **Visual**: Scroll down slightly to display the "Workflow Steps" section.
*   **Voice**: *"Standard credit scoring algorithms output uncalibrated risk ratings without explanatory reasoning. CrediSense AI bridges this gap by combining tuned gradient boosted tree pipelines, isotonic probability calibration, and local SHAP explanations to support underwriters."*

### **0:50–1:20 | Homepage & Project Metrics**
*   **Visual**: Point to the "Metrics" container showing 74 Tests and Isotonic Calibration.
*   **Voice**: *"Here we see our project statistics. The system utilizes 9 applicant-time inputs. The pipeline contains comprehensive validation and is verified by an automated test suite containing 74 tests."*

### **1:20–2:20 | Live Applicant Prediction (Demonstration)**
*   **Visual**: Fill in the Low Risk profile inputs (Age 35, Income $120k, mortgage, requested $10k, LTI auto-calculates to 0.0833, default N, cred length 8). Click "Assess Credit Risk".
*   **Voice**: *"Let's input a strong applicant profile. As we input annual income and loan amount, our client-side JavaScript calculates the loan-to-income ratio automatically. Clicking Assess submits the data, running it through the preprocessor pipeline and calibrated model."*

### **2:20–3:00 | Risk Probability & Categories**
*   **Visual**: Show the result dashboard, pointing to the calibrated probability of **0.47%**, the "Low Risk" badge, and the horizontal visual gauge scale.
*   **Voice**: *"The dashboard displays a calibrated default probability of 0.47%, falling within the policy-selected Low Risk boundary of under 12%. The visual gauge maps the applicant's risk position precisely on our color-coded spectrum."*

### **3:00–3:40 | SHAP Explainability Breakdown**
*   **Visual**: Point to the "Top Risk-Decreasing Factors" (Green cards showing Annual Income and LTI).
*   **Voice**: *"Down here, the SHAP panel lists log-odds contributions from the underlying XGBoost model. For this applicant, high income and low loan ratio are the primary factors reducing estimated default risk. Notice the academic disclaimer: SHAP explains model internals, not real-world causality."*

### **3:40–4:10 | System Methodology & Calibration**
*   **Visual**: Navigate to the "About System" page showing the workflow diagram.
*   **Voice**: *"On the About page, we review our pipeline flow. The system compares Plats and Isotonic calibration to align raw booster tree scores. Isotonic calibration was selected as champion, achieving a Brier Score of 0.0849."*

### **4:10–4:40 | Developer APIs & Health Checks**
*   **Visual**: Open a browser tab to `/health` showing status green, and `/api/model-info`.
*   **Voice**: *"For external integration, the backend exposes REST endpoints. The health route shows model availability, while the model-info route returns test metrics, calibration models, and pinned libraries like XGBoost 3.2.0 and SHAP 0.49.1."*

### **4:40–5:00 | Conclusion & Responsible Use**
*   **Visual**: Return to homepage form.
*   **Voice**: *"In conclusion, CrediSense AI demonstrates a robust, explainable framework for credit decision-support. It is an educational prototype and should not be used as the sole basis for real financial lending. Thank you."*
