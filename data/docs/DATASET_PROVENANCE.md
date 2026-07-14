# Dataset Provenance: Credit Risk Dataset

## General Information
- **Dataset Name**: Credit Risk Dataset (laotse version)
- **Original Source**: Kaggle Credit Risk Dataset
- **Creator/Publisher**: Uploaded to Kaggle by user "laotse" (pseudonym)
- **Download Source**: Phil Chodrow's Machine Learning course notes repository (Middlebury College Department of Computer Science)
  - **URL**: `https://raw.githubusercontent.com/PhilChodrow/ml-notes/main/data/credit-risk/credit_risk_dataset.csv`
- **Access Date**: July 14, 2026
- **Format**: CSV (Comma-Separated Values)
- **File Size**: 1.72 MB (1,803,174 bytes)

---

## License Verification Status
- **Source Repository Status**: The Middlebury College educational repository hosts the dataset for course use but does not contain a license text file.
- **Kaggle Host Status**: The widely referenced source page on Kaggle (uploaded by user "laotse") is self-labeled under the **CC0: Public Domain** license.
- **Verification Limitation**: While the Kaggle publisher self-labeled the data as CC0, there is no formal legal audit or documentation verifying the exact origin of the underlying banking simulation. 
- **Operational Policy**: We accept this dataset for academic study and portfolio demonstration purposes under the assumption of the informal CC0 public domain designation and credit its educational host (Phil Chodrow).

---

## Target Definition & Semantics
- **Target Variable**: `loan_status` (integer)
- **Verified Target Class Meanings**:
  - `0`: Lower observed credit/default risk class (historically did not default).
  - `1`: Higher observed credit/default risk class (historically defaulted).
- **Project Boundary Note**: The model maps inputs to observed risk classes. The application may translate these classifications into decision-support suggestions, but it must not be represented as reproducing actual lender approval decisions, which involve additional criteria (e.g., pricing margins, liquidity requirements, policy rules).

---

## Schema and Feature Descriptions

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **person_age** | `int64` | Age of the applicant in years. |
| **person_income** | `int64` | Annual income of the applicant in USD. |
| **person_home_ownership** | `object` (Categorical) | Home ownership status (RENT, OWN, MORTGAGE, OTHER). |
| **person_emp_length** | `float64` | Employment length of the applicant in years. |
| **loan_intent** | `object` (Categorical) | The purpose of the loan (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION). |
| **loan_grade** | `object` (Categorical) | Internal risk grade assigned to the loan (A, B, C, D, E, F, G). |
| **loan_amnt** | `int64` | The loan amount requested in USD. |
| **loan_int_rate** | `float64` | The interest rate of the loan in percentage. |
| **loan_percent_income** | `float64` | The proportion of the applicant's annual income represented by the loan amount (loan_amnt / person_income). |
| **cb_person_default_on_file** | `object` (Categorical) | Whether the applicant has a history of default on file (Y, N). |
| **cb_person_cred_hist_length** | `int64` | The length of the applicant's credit history in years. |
| **loan_status** | `int64` (Target) | Binary loan default status (0 = Lower risk, 1 = Higher risk). |

---

## Known Limitations & Risks

1.  **Simulated Nature**: The dataset is synthetically generated. Some correlations might not reflect real-world economic interactions.
2.  **Data Quality Issues**:
    *   **Outliers/Impossible Values**: Contains impossible ages (e.g. age of 144) and impossible employment lengths (e.g. 123 years of employment).
    *   **Logical Inconsistencies**: Instances where employment length is greater than (age - 14).
    *   **Missing Values**: 9.56% of interest rates and 2.75% of employment lengths are missing and must be handled during preprocessing.
3.  **No Temporal Dimension**: Lacks timestamps or date fields, meaning macroeconomic shifts (e.g., changes in central bank interest rates, recessions) cannot be modeled or analyzed.
4.  **Target Proxy**: The target is a default indicator (`loan_status`). In automated systems, default risk is only one component of credit approval decisions.
5.  **Fairness Risks**:
    *   `person_age` is present. Age discrimination in credit scoring is legally restricted under fair lending acts (e.g. Equal Credit Opportunity Act - ECOA). In this project, age must be monitored for technical bias (e.g., assessing prediction error disparities across age brackets) while maintaining human underwriters in the loop.
