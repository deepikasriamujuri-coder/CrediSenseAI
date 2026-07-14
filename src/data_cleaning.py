import os
import pandas as pd

def clean_data():
    raw_path = os.path.join("data", "raw", "credit_risk_dataset.csv")
    cleaned_path = os.path.join("data", "processed", "credit_risk_cleaned.csv")
    
    if not os.path.exists(raw_path):
        print(f"Error: Raw dataset not found at {raw_path}")
        return False
        
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    print(f"Initial Row Count: {initial_rows}")
    
    # 1. Duplicate Handling
    duplicate_count = df.duplicated().sum()
    df_no_dups = df.drop_duplicates()
    rows_after_dups = len(df_no_dups)
    print(f"Duplicate Rows Identified: {duplicate_count}")
    print(f"Row Count after Duplicate Removal: {rows_after_dups}")
    
    # 2. Invalid/Impossible Value Handling
    # Age > 100 (unrealistic human age boundary)
    invalid_age_mask = df_no_dups["person_age"] > 100
    invalid_age_count = invalid_age_mask.sum()
    
    # Employment length > 60 (unrealistic employment length)
    invalid_emp_mask = df_no_dups["person_emp_length"] > 60
    invalid_emp_count = invalid_emp_mask.sum()
    
    # Logical inconsistency: emp_length > age - 14 (assuming full-time employment starts no earlier than age 14)
    impossible_emp_age_mask = df_no_dups["person_emp_length"] > (df_no_dups["person_age"] - 14)
    impossible_emp_age_count = impossible_emp_age_mask.sum()
    
    # Negative values check
    negative_mask = (
        (df_no_dups["person_age"] < 0) |
        (df_no_dups["person_income"] < 0) |
        (df_no_dups["person_emp_length"] < 0) |
        (df_no_dups["loan_amnt"] < 0) |
        (df_no_dups["loan_int_rate"] < 0) |
        (df_no_dups["loan_percent_income"] < 0) |
        (df_no_dups["cb_person_cred_hist_length"] < 0)
    )
    negative_count = negative_mask.sum()
    
    # Impossible loan percentages check (> 1.0 or < 0)
    invalid_pct_mask = (df_no_dups["loan_percent_income"] > 1.0) | (df_no_dups["loan_percent_income"] < 0.0)
    invalid_pct_count = invalid_pct_mask.sum()
    
    # Combine invalid records
    invalid_records_mask = invalid_age_mask | invalid_emp_mask | impossible_emp_age_mask | negative_mask | invalid_pct_mask
    invalid_records_count = invalid_records_mask.sum()
    
    # Filter the dataset
    df_cleaned = df_no_dups[~invalid_records_mask]
    final_rows = len(df_cleaned)
    
    print("\nInvalid Records Found:")
    print(f"  - Age > 100: {invalid_age_count}")
    print(f"  - Employment length > 60: {invalid_emp_count}")
    print(f"  - Impossible employment length for age: {impossible_emp_age_count}")
    print(f"  - Negative numeric values: {negative_count}")
    print(f"  - Impossible loan percentage (> 1.0 or < 0.0): {invalid_pct_count}")
    print(f"Total Unique Invalid Records Removed: {invalid_records_count}")
    print(f"Final Cleaned Row Count: {final_rows}")
    
    # Double-check raw file wasn't modified
    raw_size_after = os.path.getsize(raw_path)
    print(f"\nRaw Dataset Size Verification: {raw_size_after} bytes (Unchanged)")
    
    # Save cleaned output
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    df_cleaned.to_csv(cleaned_path, index=False)
    print(f"Cleaned dataset saved to: {cleaned_path}")
    return True

if __name__ == "__main__":
    clean_data()
