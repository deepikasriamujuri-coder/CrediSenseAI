import os
import pandas as pd
import numpy as np

def run_analysis():
    raw_path = os.path.join("data", "raw", "credit_risk_dataset.csv")
    if not os.path.exists(raw_path):
        print(f"Error: Raw dataset not found at {raw_path}")
        return
        
    df = pd.read_csv(raw_path)
    
    print("=== CrediSense AI Initial Data Analysis ===")
    
    # Dimensions
    print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print("")
    
    # Columns & Data Types
    print("Schema & Data Types:")
    for col in df.columns:
        print(f"  - {col:<30}: {str(df[col].dtype):<10} | Non-Null Count: {df[col].count()}")
    print("")
    
    # Missing Values
    print("Missing Values Per Column:")
    missing = df.isnull().sum()
    for col, val in missing.items():
        if val > 0:
            pct = (val / len(df)) * 100
            print(f"  - {col:<30}: {val:<6} ({pct:.2f}%)")
    if missing.sum() == 0:
        print("  - No missing values found.")
    print("")
    
    # Duplicate Rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate Rows Count: {duplicates} ({ (duplicates / len(df)) * 100:.2f}%)")
    print("")
    
    # Target Analysis
    target_col = "loan_status"
    if target_col in df.columns:
        counts = df[target_col].value_counts()
        percentages = df[target_col].value_counts(normalize=True) * 100
        print("Target Variable Distribution (loan_status):")
        print("  According to database context, 1 = Default (rejected), 0 = Non-default (approved)")
        for idx in counts.index:
            meaning = "Default (Rejected / High Risk)" if idx == 1 else "Non-default (Approved / Low Risk)"
            print(f"  - Class {idx} ({meaning}): {counts[idx]:<6} ({percentages[idx]:.2f}%)")
    else:
        print(f"Warning: Target column '{target_col}' not found.")
    print("")
    
    # Unique values for Categorical Columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print("Categorical Columns Unique Values:")
    for col in categorical_cols:
        unique_vals = df[col].unique()
        print(f"  - {col:<30}: {len(unique_vals)} unique values -> {list(unique_vals)}")
    print("")
    
    # Descriptive stats for Numerical Columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude loan_status from statistics summary
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
        
    print("Descriptive Statistics for Numerical Columns:")
    desc = df[numerical_cols].describe().T
    print(desc.to_string())
    print("")
    
    # Suspicious or Impossible Values
    print("Suspicious / Impossible Values Inspection:")
    # Age checks
    suspicious_age = df[df["person_age"] > 100]
    print(f"  - Applicants with Age > 100: {len(suspicious_age)} (Max age found: {df['person_age'].max()})")
    # Employment length checks
    suspicious_emp = df[df["person_emp_length"] > 60]
    print(f"  - Applicants with Employment Length > 60 years: {len(suspicious_emp)} (Max emp length found: {df['person_emp_length'].max()})")
    # Employment length vs Age checks
    impossible_emp_age = df[df["person_emp_length"] > (df["person_age"] - 14)]
    print(f"  - Applicants where Employment Length is impossible given Age (emp_length > age - 14): {len(impossible_emp_age)}")
    # Interest rate check
    print(f"  - Loan Interest Rate range: {df['loan_int_rate'].min()}% to {df['loan_int_rate'].max()}%")
    print("")
    
    # Outliers (using Interquartile Range for extreme outliers)
    print("Outlier Observations (IQR Method - 1.5 * IQR):")
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        pct_out = (len(outliers) / len(df)) * 100
        print(f"  - {col:<30}: {len(outliers):<6} ({pct_out:.2f}%) [Bounds: {lower_bound:.2f} to {upper_bound:.2f}]")
    print("")
    
    # Target Leakage Assessment
    print("Target Leakage Risks:")
    print("  - 'loan_grade': Represents risk ranking assigned by credit scoring. Highly correlated with default risk. Needs careful evaluation if this information is generated *post* application.")
    print("  - 'loan_int_rate': Interest rates are often set based on the applicant's risk rating (loan grade) which is a proxy for default risk. Needs to be inspected for leakage.")
    print("  - 'loan_percent_income': Ratio of loan_amnt to person_income. Simple mathematical combination of features, not a leakage risk, but redundant.")
    print("")
    
    # Fairness-Sensitive Attributes
    print("Fairness-Sensitive Feature Analysis:")
    print("  - 'person_age': Age is present in the dataset. Under fair lending laws (e.g., ECOA), using age directly in credit scoring is highly restricted to avoid age discrimination. We must assess metric parity across age cohorts in later phases.")
    print("  - No race, gender, marital status, or nationality features are present in this dataset, which reduces the profile of direct legal discrimination risks.")

if __name__ == "__main__":
    run_analysis()
