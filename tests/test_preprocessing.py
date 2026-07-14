import unittest
import os
import pandas as pd
import numpy as np
from src.preprocessing import get_preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
from src.data_cleaning import clean_data

class TestPreprocessingAndCleaning(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.raw_path = os.path.join("data", "raw", "credit_risk_dataset.csv")
        cls.cleaned_path = os.path.join("data", "processed", "credit_risk_cleaned.csv")
        
        # Verify raw exists. If not, download or error
        if not os.path.exists(cls.raw_path):
            raise FileNotFoundError("Raw dataset must be downloaded before running tests.")
            
        cls.raw_df = pd.read_csv(cls.raw_path)
        cls.cleaned_df = pd.read_csv(cls.cleaned_path)
        
    def test_raw_data_unchanged(self):
        """Verify raw data file remains completely unmodified."""
        raw_df_now = pd.read_csv(self.raw_path)
        # Check equality of dimensions and row content hash/values
        self.assertEqual(raw_df_now.shape, (32581, 12))
        pd.testing.assert_frame_equal(self.raw_df, raw_df_now)
        
    def test_duplicate_removal(self):
        """Verify exact duplicate rows are identified and removed."""
        # Cleaned dataset should contain no duplicates
        self.assertEqual(self.cleaned_df.duplicated().sum(), 0)
        # Initial raw duplicates count was 165
        raw_dups = self.raw_df.duplicated().sum()
        self.assertEqual(raw_dups, 165)
        
    def test_invalid_age_handling(self):
        """Verify that ages > 100 are filtered out."""
        # Cleaned dataset should not have ages > 100
        invalid_ages = self.cleaned_df[self.cleaned_df["person_age"] > 100]
        self.assertEqual(len(invalid_ages), 0)
        
    def test_impossible_employment_length_handling(self):
        """Verify that physically impossible employment lengths are filtered out."""
        # Cleaned dataset should not have employment lengths > 60
        invalid_emp_len = self.cleaned_df[self.cleaned_df["person_emp_length"] > 60]
        self.assertEqual(len(invalid_emp_len), 0)
        
        # Cleaned dataset should not have employment length > age - 14
        impossible_emp_age = self.cleaned_df[self.cleaned_df["person_emp_length"] > (self.cleaned_df["person_age"] - 14)]
        self.assertEqual(len(impossible_emp_age), 0)
        
    def test_feature_schema_validation(self):
        """Verify all planned features are present in the cleaned schema."""
        for col in NUMERICAL_FEATURES:
            self.assertIn(col, self.cleaned_df.columns)
            
        for col in CATEGORICAL_FEATURES:
            self.assertIn(col, self.cleaned_df.columns)
            
    def test_target_validation(self):
        """Verify target column exists and contains only binary risk classes."""
        self.assertIn(TARGET_COLUMN, self.cleaned_df.columns)
        unique_targets = sorted(self.cleaned_df[TARGET_COLUMN].unique())
        self.assertEqual(unique_targets, [0, 1])
        
    def test_preprocessing_pipeline_creation(self):
        """Verify that ColumnTransformer is constructed with proper numerical and categorical sub-pipelines."""
        preprocessor = get_preprocessor()
        
        # Verify transformer count and name groupings
        transformers = preprocessor.transformers
        self.assertEqual(len(transformers), 2)
        
        # Check num transformer properties
        self.assertEqual(transformers[0][0], "num")
        self.assertEqual(transformers[0][2], NUMERICAL_FEATURES)
        
        # Check cat transformer properties
        self.assertEqual(transformers[1][0], "cat")
        self.assertEqual(transformers[1][2], CATEGORICAL_FEATURES)
        
        # Verify simple fit-transform on a small slice of cleaned data doesn't throw errors
        test_df = self.cleaned_df.head(20).copy()
        # Create dummy target
        y_dummy = test_df[TARGET_COLUMN]
        X_dummy = test_df.drop(columns=[TARGET_COLUMN])
        
        # Run fit transform
        X_transformed = preprocessor.fit_transform(X_dummy, y_dummy)
        # Expected columns: 6 numerical + one-hot encoded categories
        self.assertGreater(X_transformed.shape[1], 6)
        self.assertEqual(X_transformed.shape[0], 20)

if __name__ == "__main__":
    unittest.main()
