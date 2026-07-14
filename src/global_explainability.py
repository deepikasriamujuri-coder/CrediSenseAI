import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Monkeypatch SHAP to handle XGBoost 3.x base_score bracket format
import shap.explainers._tree as shap_tree
from shap.explainers.other._ubjson import decode_ubjson_buffer

_original_decode_ubjson = decode_ubjson_buffer

def _patched_decode_ubjson(*args, **kwargs):
    result = _original_decode_ubjson(*args, **kwargs)
    try:
        if "learner" in result:
            learner = result["learner"]
            if "learner_model_param" in learner:
                params = learner["learner_model_param"]
                if "base_score" in params:
                    val = str(params["base_score"])
                    if val.startswith("[") and val.endswith("]"):
                        params["base_score"] = val[1:-1]
    except Exception:
        pass
    return result

shap_tree.decode_ubjson_buffer = _patched_decode_ubjson

from src.prediction import CreditPredictor
from src.preprocessing import get_preprocessor, TARGET_COLUMN

def run_global_explainability():
    cleaned_path = os.path.join("data", "processed", "credit_risk_cleaned.csv")
    eval_dir = os.path.join("models", "evaluation")
    shap_dir = os.path.join("static", "images", "shap")
    example_shap_dir = os.path.join("static", "images", "shap", "examples")
    
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(shap_dir, exist_ok=True)
    os.makedirs(example_shap_dir, exist_ok=True)
    
    if not os.path.exists(cleaned_path):
        print(f"Error: Cleaned dataset not found at {cleaned_path}")
        return False
        
    df = pd.read_csv(cleaned_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Train / Test split (preserve test split isolation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Load calibrated ensemble model
    predictor = CreditPredictor()
    model = predictor.model
    estimators = [clf.estimator for clf in model.calibrated_classifiers_]
    
    # Sample a representative subset of the training data (500 samples)
    sample_size = 500
    X_sample = X_train.sample(n=sample_size, random_state=42)
    
    # We recover feature names and build maps
    first_preprocessor = estimators[0].named_steps["preprocessor"]
    from src.preprocessing import get_feature_names
    transformed_features = get_feature_names(first_preprocessor)
    
    # Map transformed names to friendly labels for plots
    friendly_names = {
        "person_age": "Age",
        "person_income": "Annual Income",
        "person_emp_length": "Employment Duration",
        "loan_amnt": "Loan Amount",
        "loan_percent_income": "Loan-to-Income Ratio",
        "cb_person_cred_hist_length": "Credit History Length",
        "person_home_ownership": "Home Ownership",
        "loan_intent": "Loan Purpose",
        "cb_person_default_on_file": "Previous Default"
    }
    
    def map_to_friendly(tech_name):
        if tech_name.startswith("num__"):
            base = tech_name.replace("num__", "")
            return friendly_names.get(base, base)
        elif tech_name.startswith("cat__"):
            base = tech_name.replace("cat__", "")
            for cat_field in ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]:
                if base.startswith(cat_field + "_"):
                    category_value = base.replace(cat_field + "_", "")
                    if category_value == "Y": category_value = "Yes"
                    if category_value == "N": category_value = "No"
                    return f"{friendly_names.get(cat_field, cat_field)}: {category_value}"
        return tech_name
        
    friendly_feature_labels = [map_to_friendly(f) for f in transformed_features]
    
    # Preprocess sample data using the first fold's preprocessor
    X_sample_trans = first_preprocessor.transform(X_sample)
    
    # Generate SHAP matrix for the sample by averaging across all 5 estimators
    all_shap_matrices = []
    all_base_values = []
    
    for i, pipeline in enumerate(estimators):
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        
        # Preprocess using the split's specific preprocessor
        X_trans_fold = preprocessor.transform(X_sample)
        
        explainer = shap.TreeExplainer(classifier)
        shap_res = explainer(X_trans_fold)
        
        shap_vals = shap_res.values
        base_vals = shap_res.base_values
        
        all_shap_matrices.append(shap_vals)
        all_base_values.append(base_vals)
        
    avg_shap_values = np.mean(all_shap_matrices, axis=0)
    
    # 1. Plot Beeswarm Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(avg_shap_values, X_sample_trans, feature_names=friendly_feature_labels, show=False)
    plt.title("CrediSense AI: Global SHAP Summary Beeswarm (500 Train Samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "shap_summary_plot.png"), dpi=200)
    plt.close()
    
    # 2. Plot Bar Importance Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(avg_shap_values, X_sample_trans, feature_names=friendly_feature_labels, plot_type="bar", show=False)
    plt.title("CrediSense AI: Global SHAP Feature Importance (Bar Plot)")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "shap_bar_importance.png"), dpi=200)
    plt.close()
    
    print("Saved global SHAP plots successfully.")
    
    # 3. Compare Built-In vs Mean Absolute SHAP Importance
    all_importances = []
    for pipeline in estimators:
        classifier = pipeline.named_steps["classifier"]
        all_importances.append(classifier.feature_importances_)
        
    avg_built_in_importance = np.mean(all_importances, axis=0)
    
    # Compute mean absolute SHAP importance
    mean_abs_shap = np.mean(np.abs(avg_shap_values), axis=0)
    
    # Save comparison as CSV
    df_imp = pd.DataFrame({
        "Technical_Feature": transformed_features,
        "Friendly_Feature": friendly_feature_labels,
        "XGBoost_BuiltIn_Importance": avg_built_in_importance,
        "Mean_Absolute_SHAP_Importance": mean_abs_shap
    }).sort_values(by="Mean_Absolute_SHAP_Importance", ascending=False)
    
    df_imp.to_csv(os.path.join(eval_dir, "feature_importance_comparison.csv"), index=False)
    print("\nFeature Importance Comparison Table:")
    print(df_imp.to_string(index=False))
    print("")
    
    # 4. Generate Example Individual Explanation Chart
    y_sample_pred = model.predict(X_sample)
    high_risk_indices = np.where(y_sample_pred == 1)[0]
    
    if len(high_risk_indices) > 0:
        target_idx = high_risk_indices[0]
        local_shap = avg_shap_values[target_idx]
        
        # Sort features by absolute contribution
        sort_idx = np.argsort(np.abs(local_shap))[::-1]
        
        top_features = [friendly_feature_labels[i] for i in sort_idx[:10]]
        top_shap_vals = [local_shap[i] for i in sort_idx[:10]]
        
        # Plot custom bar chart
        plt.figure(figsize=(8, 5))
        colors = ['#E76F51' if val > 0 else '#2A9D8F' for val in top_shap_vals]
        y_pos = np.arange(len(top_features))
        
        plt.barh(y_pos, top_shap_vals, align='center', color=colors, alpha=0.85)
        plt.yticks(y_pos, top_features)
        plt.gca().invert_yaxis()
        plt.xlabel("SHAP Value (Log-Odds Impact)")
        plt.title("Individual Prediction Explanation (High Risk Case Example)")
        plt.axvline(0, color='gray', linestyle='-', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(example_shap_dir, "individual_waterfall.png"), dpi=200)
        plt.close()
        print("Saved example individual explanation plot successfully.")
    else:
        print("No high risk applicant found in sample to generate example chart.")
        
    return True

if __name__ == "__main__":
    run_global_explainability()
