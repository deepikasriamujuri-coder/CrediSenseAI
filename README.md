# CrediSense AI: Explainable Credit Risk Prediction System

CrediSense AI is an academic decision-support platform built as a final-year B.Tech machine learning project. It predicts credit default risk, calibrates output default probabilities, categorizes applicants, and provides local explainability using SHAP.

---

## 1. Project Overview
Evaluating credit eligibility is a critical task in retail banking. However, gradient-boosted models output raw ranking scores that do not align with actual default frequencies, and they function as opaque "black boxes." 

**CrediSense AI** solves these issues by:
- Preprocessing applicant profiles inside isolated pipelines.
- Predicting defaults using a tuned ensemble of **XGBoost** models.
- Calibrating output scores using **Isotonic Calibration** to ensure highly reliable probabilities.
- Generating local feature-level contributions using averaged **SHAP** values.

*Note: This is an academic prototype designed for training and educational demonstrations. It is not an automated banking decision maker and does not possess lending or loan approval authority.*

---

## 2. Key Features
- **Calibrated Credit-Risk Prediction**: Outputs empirical default probabilities.
- **Ensemble SHAP Explanations**: Local log-odds contribution weights averaged across 5 fold estimators.
- **Policy Risk Categorization**: Allocates applicants to Low, Medium, and High risk buckets.
- **Interactive UI Dashboard**: Premium dark-navy layout with real-time calculators and visual gauges.
- **Production REST APIs**: Exposes endpoints for health monitoring, model metadata, and predictions.
- **Graceful SHAP Degradation**: System degrades gracefully if explainability layers fail.
- **Server-side Consistency Verification**: Verifies loan-to-income ratios to protect data integrity.
- **Robust Test Coverage**: 74 automated tests validating preprocessors, models, and endpoints.

---

## 3. Technology Stack
- **Backend Framework**: Python 3.10 &bull; Flask 3.1.3 &bull; Gunicorn (production readiness)
- **Machine Learning Core**: Scikit-Learn 1.7.2 &bull; XGBoost 3.2.0 &bull; Joblib 1.5.3
- **Explainable AI**: SHAP 0.49.1
- **Data Preprocessing**: Pandas 2.3.3 &bull; NumPy 2.2.6
- **Data Visualization**: Matplotlib 3.10.0 &bull; Seaborn 0.13.2
- **Frontend Design**: HTML5 &bull; CSS3 (Vanilla) &bull; JavaScript (ES6)

---

## 4. System Architecture
```
Applicant Input (Form/JSON)
       │
       ▼
Input Validation (NaNs, Bounds, Ratio Check)
       │
       ▼
Preprocessing Pipeline (Robust Scaling & OHE)
       │
       ▼
Calibrated XGBoost Ensemble (5 Folds) ──► SHAP Explainer (Bagged Log-Odds)
       │                                            │
       ▼                                            ▼
Default Probability & Category             Feature Contributions & Mappings
       │                                            │
       └────────────────────┬───────────────────────┘
                            │
                            ▼
               Dashboard View / REST Response
```

---

## 5. Dataset & Production Schema
The system is trained on the Kaggle Credit Risk dataset. The raw dataset contains 32,581 records, cleaned of duplicates, impossible employment durations, and age outliers, yielding a final training pool of 32,409 rows.

The final model utilizes exactly **9 features**:
1.  `person_age` (Integer): Applicant age `[18, 100]`
2.  `person_income` (Float): Annual income `[0, inf)`
3.  `person_home_ownership` (Categorical): `{'RENT', 'OWN', 'MORTGAGE', 'OTHER'}`
4.  `person_emp_length` (Float): Employment duration in years `[0, Age - 14]`
5.  `loan_intent` (Categorical): `{'PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'}`
6.  `loan_amnt` (Float): Requested loan amount `(0, inf)`
7.  `loan_percent_income` (Float): Loan-to-income ratio `[0.0, 1.0]`
8.  `cb_person_default_on_file` (Categorical): History of default `{'Y', 'N'}`
9.  `cb_person_cred_hist_length` (Integer): Credit history length in years `[0, inf)`

---

## 6. Machine Learning Workflow
- **Model Selection**: XGBoost Weighted was selected as the champion model during training-set 5-fold cross-validation, achieving a CV PR-AUC of **0.8109** and Recall of **73.47%**.
- **Calibration**: Isotonic Calibration was chosen over Platt Scaling, reducing the Brier score on validation data by **20.9%** (to 0.0849).
- **Ensemble Strategy**: We wrap the model in a 5-fold cross-validation ensemble. Inference predictions and SHAP explainers run across all 5 folds and average their outputs.

---

## 7. Critical Threshold & Boundary Distinction
- **Binary Classification Threshold (0.50)**: Used to classify applicants into predicted binary Class 0 (Lower Risk) or Class 1 (Higher Risk).
- **Policy Risk Boundaries**: Based on empirical out-of-fold evaluations:
  - **`Low Risk`** ($P < 0.12$): Observed default rate of **4.70%** (under 5% target, suitable for automated approvals support).
  - **`Medium Risk`** ($0.12 \le P < 0.35$): Observed default rate of **23.27%** (requires manual underwriter review).
  - **`High Risk`** ($P \ge 0.35$): Observed default rate of **79.17%** (confident default risk concentration).
- An applicant with a probability of 0.40 is mapped to the High Risk category (business policy), but classified as Class 0 (verdict), because 0.40 is below the 0.50 classification threshold.

---

## 8. Explainability (SHAP)
SHAP values represent additive log-odds contribution weights in the underlying XGBoost model decision space. They do not map linearly to calibrated probability adjustments.
- **Risk-Increasing Factors**: Feature states that push risk higher (positive SHAP log-odds).
- **Risk-Decreasing Factors**: Feature states that pull risk lower (negative SHAP log-odds).
- **XGBoost 3.x Compatibility Workaround**: We monkeypatched SHAP's UBJSON parser `decode_ubjson_buffer` to prevent a float compilation crash on the bracketed `base_score` format.

---

## 9. Project Directory Tree
```
CrediSenseAI/
├── config.py                 # Environment configurations
├── app.py                    # Flask application factory and routings
├── requirements.txt          # Pinned dependency lists
├── .gitignore                # Git exclude configs
├── .env.example              # Env variable templates
├── data/
│   ├── raw/                  # Clean raw credit risk CSV
│   └── processed/            # Cleaned data
├── docs/
│   ├── API.md                # REST API endpoints specifications
│   ├── EXPLAINABILITY.md     # SHAP calibration and math details
│   ├── DEMO_GUIDE.md         # Demo profiles and scripts
│   └── VIVA_GUIDE.md         # 25 Q&A study guide
├── models/                   # Serialized classifiers and metadata
├── src/                      # Predictors, cleaners, and explainers
├── static/
│   ├── css/                  # Custom CSS stylesheets
│   ├── js/                   # Dynamics JS scripts
│   └── images/               # Global SHAP beeswarm graphs
├── templates/                # Jinja browser views
└── tests/                    # 74 automated validation tests
```

---

## 10. Installation & Run Instructions (Windows)

1.  **Clone the Repository**:
    ```powershell
    git clone [Placeholder-URL-For-Later-Replacement]
    cd CrediSenseAI
    ```
2.  **Create and Activate Virtual Environment**:
    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    ```
3.  **Install Pinned Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```
4.  **Set Up Environment File**:
    ```powershell
    copy .env.example .env
    ```
5.  **Run the Web Application**:
    ```powershell
    python app.py
    ```
6.  **Access the Dashboard**:
    Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 11. REST API Endpoint Summary
- `GET /health`: Health status.
- `GET /api/model-info`: Production test scores, boundaries, and variables.
- `POST /api/predict`: Returns calibrated default probability and classification.
- `POST /api/explain`: Returns local SHAP log-odds contribution weights.
- `POST /api/predict-with-explanation`: Combined endpoint (removes duplicate inference).

For JSON schemas, reference [docs/API.md](file:///C:/Users/anilk/Documents/CrediSenseAI/docs/API.md).

---

## 12. Running the Test Suite
To execute all 74 automated tests (validating preprocessors, tuners, predictors, SHAP math, and API codes):
```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## 13. Limitations & Disclaimers
- **Tabular Scope**: The model is restricted to structured financial applications.
- **No Causal SHAP**: SHAP explains model internals, not real-world causality.
- **Academic Disclaimer**: CrediSense AI is designed as a university prototype. It should not be used as the sole basis for real banking credit decisions.
