import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Define feature configuration
# Columns "loan_grade" and "loan_int_rate" are classified as potential post-assessment features.
# Because they are potentially unavailable at the intended initial application prediction time, 
# they are excluded from the primary application-time model to align with a realistic lending workflow.
EXCLUDED_FEATURES = ["loan_grade", "loan_int_rate"]
TARGET_COLUMN = "loan_status"

NUMERICAL_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file"
]

# Numerical and Categorical feature lists including potential post-assessment features
FULL_NUMERICAL_FEATURES = NUMERICAL_FEATURES + ["loan_int_rate"]
FULL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + ["loan_grade"]

def get_preprocessor():
    """
    Creates and returns an unfitted Scikit-learn ColumnTransformer for the primary application-time feature set.
    - Numerical: Median Imputer followed by RobustScaler.
    - Categorical: Most Frequent Imputer followed by OneHotEncoder with handle_unknown='ignore'.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop" # Drop any other columns (e.g. loan_grade, loan_int_rate, loan_status if present)
    )
    
    return preprocessor

def get_full_preprocessor():
    """
    Creates and returns an unfitted Scikit-learn ColumnTransformer for the Full Feature Benchmark.
    Includes "loan_grade" and "loan_int_rate".
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FULL_NUMERICAL_FEATURES),
            ("cat", categorical_transformer, FULL_CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    
    return preprocessor

def get_feature_names(column_transformer):
    """
    Extracts the feature names from a fitted ColumnTransformer.
    Useful for mapping SHAP output values back to original feature names.
    """
    feature_names = []
    
    # Loop over transformers
    for name, trans, cols in column_transformer.transformers_:
        if name == "remainder" or trans == "drop":
            continue
            
        if isinstance(trans, Pipeline):
            # Resolve encoder if present
            encoder = trans.named_steps.get("encoder", None)
            if encoder and isinstance(encoder, OneHotEncoder):
                # Get one-hot names
                ohe_cols = encoder.get_feature_names_out(cols)
                feature_names.extend(ohe_cols)
            else:
                feature_names.extend(cols)
        elif isinstance(trans, OneHotEncoder):
            ohe_cols = trans.get_feature_names_out(cols)
            feature_names.extend(ohe_cols)
        else:
            feature_names.extend(cols)
            
    return feature_names

if __name__ == "__main__":
    # Test preprocessor instantiation
    preprocessor = get_preprocessor()
    full_preprocessor = get_full_preprocessor()
    print("Primary Preprocessor:")
    print(preprocessor)
    print("\nFull Preprocessor:")
    print(full_preprocessor)
