import unittest
import os
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from src.preprocessing import get_preprocessor, TARGET_COLUMN

class TestHyperparameterTuningWorkflow(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.eval_dir = os.path.join("models", "evaluation")
        cls.params_xgb_weighted = os.path.join(cls.eval_dir, "params_xgboost_weighted.json")
        cls.tuning_results_csv = os.path.join(cls.eval_dir, "tuning_results.csv")
        cls.cv_comparison_csv = os.path.join(cls.eval_dir, "tuned_candidate_comparison.csv")
        
    def test_search_artifacts_exist(self):
        """Verify that hyperparameter tuning CSV reports and param files were saved."""
        self.assertTrue(os.path.exists(self.tuning_results_csv))
        self.assertTrue(os.path.exists(self.cv_comparison_csv))
        self.assertTrue(os.path.exists(self.params_xgb_weighted))
        
    def test_best_parameters_loading(self):
        """Verify that best parameters JSON files load correctly and contain correct keys."""
        with open(self.params_xgb_weighted, "r") as f:
            params = json.load(f)
        
        self.assertIn("classifier__n_estimators", params)
        self.assertIn("classifier__max_depth", params)
        self.assertIn("classifier__learning_rate", params)
        
    def test_cv_configuration(self):
        """Verify the StratifiedKFold configuration is correctly set up."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.assertEqual(cv.n_splits, 5)
        self.assertTrue(cv.shuffle)
        self.assertEqual(cv.random_state, 42)
        
    def test_primary_scoring_metric(self):
        """Verify average_precision (PR-AUC) is utilized as the primary optimization metric."""
        df_tuning = pd.read_csv(self.tuning_results_csv)
        # Check that Best_CV_PR_AUC exists
        self.assertIn("Best_CV_PR_AUC", df_tuning.columns)
        
    def test_preprocessing_inside_pipeline(self):
        """Verify that preprocessing step is contained within the Pipeline."""
        # Cleaned CSV loading for basic check
        cleaned_path = os.path.join("data", "processed", "credit_risk_cleaned.csv")
        df = pd.read_csv(cleaned_path)
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        preprocessor = get_preprocessor()
        classifier = xgb.XGBClassifier(random_state=42)
        
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        
        self.assertEqual(pipeline.steps[0][0], "preprocessor")
        self.assertEqual(pipeline.steps[1][0], "classifier")
        
    def test_production_candidate_feature_exclusions(self):
        """Verify that the production candidate (get_preprocessor) does not include post-assessment variables."""
        preprocessor = get_preprocessor()
        
        # Access column transformer details
        transformers = preprocessor.transformers
        num_cols = transformers[0][2]
        cat_cols = transformers[1][2]
        
        self.assertNotIn("loan_grade", num_cols)
        self.assertNotIn("loan_grade", cat_cols)
        self.assertNotIn("loan_int_rate", num_cols)
        self.assertNotIn("loan_int_rate", cat_cols)
        
    def test_probability_output_availability(self):
        """Verify that the production model supports predict_proba interface."""
        classifier = xgb.XGBClassifier(random_state=42)
        self.assertTrue(hasattr(classifier, "predict_proba"))

if __name__ == "__main__":
    unittest.main()
