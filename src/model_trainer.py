import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.preprocessing import get_preprocessor, get_full_preprocessor, TARGET_COLUMN

def run_experiment():
    cleaned_path = os.path.join("data", "processed", "credit_risk_cleaned.csv")
    eval_dir = os.path.join("models", "evaluation")
    img_eval_dir = os.path.join("static", "images", "model_evaluation")
    
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(img_eval_dir, exist_ok=True)
    
    if not os.path.exists(cleaned_path):
        print(f"Error: Cleaned dataset not found at {cleaned_path}")
        return False
        
    df = pd.read_csv(cleaned_path)
    
    # Target and Features separation
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # 1. Dataset Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("=== Dataset Splitting Summary ===")
    print(f"Train Set Shape: {X_train.shape[0]} rows, {X_train.shape[1]} columns")
    print(f"Test Set Shape: {X_test.shape[0]} rows, {X_test.shape[1]} columns")
    print("Train Target Distribution:")
    print(f"  - Class 0 (Lower Risk): {sum(y_train == 0)} ({100*sum(y_train == 0)/len(y_train):.2f}%)")
    print(f"  - Class 1 (Higher Risk): {sum(y_train == 1)} ({100*sum(y_train == 1)/len(y_train):.2f}%)")
    print("Test Target Distribution:")
    print(f"  - Class 0 (Lower Risk): {sum(y_test == 0)} ({100*sum(y_test == 0)/len(y_test):.2f}%)")
    print(f"  - Class 1 (Higher Risk): {sum(y_test == 1)} ({100*sum(y_test == 1)/len(y_test):.2f}%)")
    print("")
    
    # Calculate scale_pos_weight for XGBoost
    scale_weight = float(sum(y_train == 0)) / float(sum(y_train == 1))
    print(f"Calculated scale_pos_weight for XGBoost imbalance correction: {scale_weight:.4f}")
    print("")
    
    # 2. Validation CV Setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Helper to calculate average precision (PR-AUC) scoring metric inside cross_validate
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision'
    }
    
    # Define models list for primary feature set (which excludes loan_grade and loan_int_rate)
    # The get_preprocessor pipeline handles dropping the excluded features automatically
    primary_models = {
        "Dummy (Prior)": DummyClassifier(strategy="prior"),
        
        "Logistic Regression (Default)": LogisticRegression(max_iter=1000, random_state=42),
        "Logistic Regression (Balanced)": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        
        "Decision Tree (Default)": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Decision Tree (Balanced)": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        
        "Random Forest (Default)": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42),
        
        "XGBoost (Default)": xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric="logloss"),
        "XGBoost (Weighted)": xgb.XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=scale_weight, random_state=42, eval_metric="logloss")
    }
    
    cv_results_list = []
    test_results_list = []
    
    # Run Primary Experiments
    print("=== Training & Evaluating Primary Feature-Set Models ===")
    for model_name, model in primary_models.items():
        print(f"Processing: {model_name}...")
        
        # Build primary pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", get_preprocessor()),
            ("classifier", model)
        ])
        
        # Cross Validation
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        # Save CV metrics averages
        cv_res = {
            "Model": model_name,
            "CV_Accuracy_Mean": cv_scores["test_accuracy"].mean(),
            "CV_Accuracy_Std": cv_scores["test_accuracy"].std(),
            "CV_Precision_Mean": cv_scores["test_precision"].mean(),
            "CV_Recall_Mean": cv_scores["test_recall"].mean(),
            "CV_F1_Mean": cv_scores["test_f1"].mean(),
            "CV_ROC_AUC_Mean": cv_scores["test_roc_auc"].mean(),
            "CV_PR_AUC_Mean": cv_scores["test_pr_auc"].mean()
        }
        cv_results_list.append(cv_res)
        
        # Evaluate on Test Set
        pipeline.fit(X_train, y_train)
        
        # Check probability support
        has_predict_proba = hasattr(pipeline.named_steps["classifier"], "predict_proba")
        has_decision_function = hasattr(pipeline.named_steps["classifier"], "decision_function")
        
        y_pred = pipeline.predict(X_test)
        
        if has_predict_proba:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        elif has_decision_function:
            y_proba = pipeline.decision_function(X_test)
        else:
            y_proba = y_pred.astype(float)
            
        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_roc = roc_auc_score(y_test, y_proba)
        test_pr = average_precision_score(y_test, y_proba)
        
        test_res = {
            "Model": model_name,
            "Test_Accuracy": test_acc,
            "Test_Precision": test_prec,
            "Test_Recall": test_rec,
            "Test_F1": test_f1,
            "Test_ROC_AUC": test_roc,
            "Test_PR_AUC": test_pr,
            "Supports_Proba": "predict_proba" if has_predict_proba else ("decision_function" if has_decision_function else "None")
        }
        test_results_list.append(test_res)
        
        # Save specific classification reports & matrices for key primary configurations
        if model_name in ["Logistic Regression (Balanced)", "Random Forest (Default)", "XGBoost (Default)", "XGBoost (Weighted)"]:
            # Save classification report txt
            report_str = classification_report(y_test, y_pred, digits=4)
            with open(os.path.join(eval_dir, f"report_{model_name.replace(' ', '_').lower()}.txt"), "w") as f:
                f.write(f"=== {model_name} Test Set Classification Report ===\n")
                f.write(report_str)
                f.write(f"\nConfusion Matrix:\n")
                f.write(str(confusion_matrix(y_test, y_pred)))
            
            # Create Confusion Matrix Plot
            plt.figure(figsize=(5, 4))
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Lower Risk", "Higher Risk"])
            disp.plot(cmap="Blues", values_format="d")
            plt.title(f"{model_name} Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(img_eval_dir, f"cm_{model_name.replace(' ', '_').lower()}.png"), dpi=150)
            plt.close()
            
            # ROC and Precision-Recall Curves
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {test_roc:.4f})")
            plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5000)")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{model_name} ROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(img_eval_dir, f"roc_{model_name.replace(' ', '_').lower()}.png"), dpi=150)
            plt.close()
            
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            plt.figure(figsize=(6, 5))
            plt.plot(rec, prec, label=f"{model_name} (AP = {test_pr:.4f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"{model_name} Precision-Recall Curve")
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(img_eval_dir, f"pr_{model_name.replace(' ', '_').lower()}.png"), dpi=150)
            plt.close()
            
    # 3. Full Feature Benchmark (Experimental comparison with loan_grade & loan_int_rate)
    print("\n=== Training & Evaluating Full Feature Benchmark ===")
    full_models = {
        "Full Feature: Logistic Regression (Balanced)": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Full Feature: Random Forest (Default)": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
        "Full Feature: XGBoost (Default)": xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric="logloss"),
        "Full Feature: XGBoost (Weighted)": xgb.XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=scale_weight, random_state=42, eval_metric="logloss")
    }
    
    feature_comparison_list = []
    
    for model_name, model in full_models.items():
        print(f"Processing Benchmark: {model_name}...")
        
        # Build full benchmark pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", get_full_preprocessor()),
            ("classifier", model)
        ])
        
        # Cross Validation
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        # Test Set
        pipeline.fit(X_train, y_train)
        has_predict_proba = hasattr(pipeline.named_steps["classifier"], "predict_proba")
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if has_predict_proba else y_pred.astype(float)
        
        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_roc = roc_auc_score(y_test, y_proba)
        test_pr = average_precision_score(y_test, y_proba)
        
        benchmark_res = {
            "Model": model_name,
            "CV_Accuracy_Mean": cv_scores["test_accuracy"].mean(),
            "CV_ROC_AUC_Mean": cv_scores["test_roc_auc"].mean(),
            "CV_PR_AUC_Mean": cv_scores["test_pr_auc"].mean(),
            "Test_Accuracy": test_acc,
            "Test_Precision": test_prec,
            "Test_Recall": test_rec,
            "Test_F1": test_f1,
            "Test_ROC_AUC": test_roc,
            "Test_PR_AUC": test_pr
        }
        feature_comparison_list.append(benchmark_res)
        
    # Save primary model comparison tables
    df_cv_res = pd.DataFrame(cv_results_list)
    df_test_res = pd.DataFrame(test_results_list)
    df_primary_comparison = pd.merge(df_cv_res, df_test_res, on="Model")
    df_primary_comparison.to_csv(os.path.join(eval_dir, "model_comparison.csv"), index=False)
    print(f"Primary comparison CSV saved to: {os.path.join(eval_dir, 'model_comparison.csv')}")
    
    # Save feature set comparisons
    df_benchmark = pd.DataFrame(feature_comparison_list)
    df_benchmark.to_csv(os.path.join(eval_dir, "feature_set_comparison.csv"), index=False)
    print(f"Benchmark comparison CSV saved to: {os.path.join(eval_dir, 'feature_set_comparison.csv')}")
    
    print("\nExperiment run completed successfully!")
    return True

if __name__ == "__main__":
    run_experiment()
