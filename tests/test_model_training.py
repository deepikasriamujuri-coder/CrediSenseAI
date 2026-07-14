import unittest
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb

from src.preprocessing import get_preprocessor, get_full_preprocessor, TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.model_trainer import run_experiment

class TestModelTrainingWorkflow(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.cleaned_path = os.path.join("data", "processed", "credit_risk_cleaned.csv")
        if not os.path.exists(cls.cleaned_path):
            raise FileNotFoundError("Cleaned dataset not found. Run cleaning script first.")
            
        cls.df = pd.read_csv(cls.cleaned_path)
        cls.X = cls.df.drop(columns=[TARGET_COLUMN])
        cls.y = cls.df[TARGET_COLUMN]
        
        # Split data
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.20, random_state=42, stratify=cls.y
        )
        
    def test_split_stratification(self):
        """Verify train/test split maintains target class stratification."""
        train_ratio = sum(self.y_train == 1) / len(self.y_train)
        test_ratio = sum(self.y_test == 1) / len(self.y_test)
        overall_ratio = sum(self.y == 1) / len(self.y)
        
        # Tolerances within 0.1%
        self.assertAlmostEqual(train_ratio, overall_ratio, places=3)
        self.assertAlmostEqual(test_ratio, overall_ratio, places=3)
        self.assertAlmostEqual(train_ratio, test_ratio, places=3)
        
    def test_no_target_in_features(self):
        """Verify target column is completely excluded from feature matrices."""
        self.assertNotIn(TARGET_COLUMN, self.X_train.columns)
        self.assertNotIn(TARGET_COLUMN, self.X_test.columns)
        
    def test_primary_preprocessor_exclusions(self):
        """Verify that primary preprocessor outputs correct dimensions, dropping excluded columns."""
        preprocessor = get_preprocessor()
        
        # Fit-transform train partition to check shape
        X_trans = preprocessor.fit_transform(self.X_train, self.y_train)
        
        # Primary preprocessor features:
        # Numeric: age, income, emp_len, loan_amnt, percent_income, cred_hist (6 columns)
        # Categorical: home_ownership (4 encoded), intent (6 encoded), default_on_file (2 encoded)
        # Total encoded categorical = 12. 6 + 12 = 18 features outputted.
        self.assertEqual(X_trans.shape[1], 18)
        
    def test_full_benchmark_preprocessor_inclusions(self):
        """Verify that full benchmark preprocessor includes additional columns."""
        preprocessor = get_full_preprocessor()
        
        X_trans = preprocessor.fit_transform(self.X_train, self.y_train)
        
        # Full preprocessor features:
        # Numeric: age, income, emp_len, loan_amnt, percent_income, cred_hist, PLUS loan_int_rate (7 columns)
        # Categorical: home_ownership (4), intent (6), default_on_file (2), PLUS loan_grade (7 encoded)
        # Total columns should be 7 + 19 = 26 features outputted.
        self.assertEqual(X_trans.shape[1], 26)
        
    def test_pipeline_construction(self):
        """Verify classifier pipelines are constructed correctly."""
        preprocessor = get_preprocessor()
        classifier = xgb.XGBClassifier(random_state=42)
        
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        
        self.assertEqual(pipeline.steps[0][0], "preprocessor")
        self.assertEqual(pipeline.steps[1][0], "classifier")
        self.assertIsInstance(pipeline.named_steps["classifier"], xgb.XGBClassifier)
        
    def test_xgboost_weighted_configuration(self):
        """Verify scale_pos_weight is passed correctly to weighted XGBoost."""
        scale_weight = float(sum(self.y_train == 0)) / float(sum(self.y_train == 1))
        model = xgb.XGBClassifier(scale_pos_weight=scale_weight, random_state=42)
        
        self.assertAlmostEqual(model.get_params()["scale_pos_weight"], scale_weight, places=4)
        
    def test_metric_calculations(self):
        """Verify metrics calculations yield float values within [0,1]."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        
        self.assertTrue(0.0 <= acc <= 1.0)
        self.assertTrue(0.0 <= prec <= 1.0)
        self.assertTrue(0.0 <= rec <= 1.0)
        self.assertEqual(acc, 0.8)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 0.5)

    def test_preprocessing_isolation(self):
        """Verify that preprocessing fitting is isolated and does not mutate raw data frames."""
        preprocessor = get_preprocessor()
        X_slice = self.X_train.head(50).copy()
        
        # Record pre-fit state
        orig_mean = X_slice["person_income"].mean()
        
        # Fit preprocessor
        preprocessor.fit(X_slice, self.y_train.head(50))
        
        # Ensure original dataframe has not been modified
        self.assertEqual(X_slice["person_income"].mean(), orig_mean)

if __name__ == "__main__":
    unittest.main()
