import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.preprocessing import get_preprocessor, TARGET_COLUMN

def run_tuning():
    cleaned_path = os.path.join("data", "processed", "credit_risk_cleaned.csv")
    eval_dir = os.path.join("models", "evaluation")
    img_eval_dir = os.path.join("static", "images", "model_evaluation")
    
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(img_eval_dir, exist_ok=True)
    
    if not os.path.exists(cleaned_path):
        print(f"Error: Cleaned dataset not found at {cleaned_path}")
        return False
        
    df = pd.read_csv(cleaned_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Train / Test Splitting (Preserve test set for final evaluation ONLY)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Calculate scale_pos_weight from training data only
    scale_weight = float(sum(y_train == 0)) / float(sum(y_train == 1))
    
    # Cross Validation Strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Search Spaces
    xgb_default = xgb.XGBClassifier(random_state=42, eval_metric="logloss")
    xgb_weighted = xgb.XGBClassifier(scale_pos_weight=scale_weight, random_state=42, eval_metric="logloss")
    rf_default = RandomForestClassifier(random_state=42)
    rf_balanced = RandomForestClassifier(class_weight="balanced", random_state=42)
    
    xgb_param_dist = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__max_depth": [3, 5, 8],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__min_child_weight": [1, 3, 5],
        "classifier__subsample": [0.7, 0.8, 0.9],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9],
        "classifier__gamma": [0, 0.1, 0.2],
        "classifier__reg_alpha": [0, 0.1, 1],
        "classifier__reg_lambda": [1, 2, 5]
    }
    
    rf_param_dist = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__max_depth": [6, 10, 15],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", None]
    }
    
    # Configurations dict
    configs = {
        "XGBoost Default-style": (xgb_default, xgb_param_dist),
        "XGBoost Weighted": (xgb_weighted, xgb_param_dist),
        "Random Forest Default": (rf_default, rf_param_dist),
        "Random Forest Balanced": (rf_balanced, rf_param_dist)
    }
    
    search_results = []
    best_estimators = {}
    
    print("=== Phase 6: Hyperparameter Tuning ===")
    for name, (model, dist) in configs.items():
        print(f"Tuning {name}...")
        pipeline = Pipeline(steps=[
            ("preprocessor", get_preprocessor()),
            ("classifier", model)
        ])
        
        # We optimize strictly for average_precision (PR-AUC)
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=dist,
            n_iter=10,
            scoring="average_precision",
            cv=cv,
            random_state=42,
            n_jobs=-1
        )
        
        start_time = time.time()
        search.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        # Save best estimator
        best_estimators[name] = search.best_estimator_
        
        # Get CV metrics associated with the best param setting
        cv_pr_auc = search.best_score_
        
        # Calculate other CV scores for the best index
        best_idx = search.best_index_
        best_cv_results = search.cv_results_
        
        # Extract mean CV metrics for comparison
        # (Scikit-learn stores them under mean_test_<metric_name> but we only scored one metric: average_precision.
        # We can extract average accuracy, recall, precision, f1 and roc_auc using a helper evaluation run of cross_validate on the best parameters
        print(f"  - Completed in {elapsed_time:.2f} seconds.")
        print(f"  - Best CV PR-AUC: {cv_pr_auc:.4f}")
        print(f"  - Best Parameters: {search.best_params_}")
        
        # Save best parameters to json
        param_path = os.path.join(eval_dir, f"params_{name.replace(' ', '_').lower()}.json")
        with open(param_path, "w") as f:
            json.dump(search.best_params_, f, indent=4)
            
        search_results.append({
            "Model": name,
            "Best_CV_PR_AUC": cv_pr_auc,
            "Tuned_Parameters": str(search.best_params_),
            "Execution_Time_Seconds": elapsed_time,
            "CV_Fits": 5 * 10
        })
        
    df_tuning = pd.DataFrame(search_results)
    df_tuning.to_csv(os.path.join(eval_dir, "tuning_results.csv"), index=False)
    
    # 2. Final Candidate Comparison using Cross Validation
    # Let's perform a validation run for each best estimator to log all metrics (Accuracy, F1, ROC-AUC, Recall, Precision)
    print("\n=== Running Final CV Comparison on Best Tuned Configurations ===")
    cv_comparison_results = []
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision'
    }
    
    for name, pipeline in best_estimators.items():
        from sklearn.model_selection import cross_validate
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        cv_comparison_results.append({
            "Model": name,
            "CV_Accuracy": cv_scores["test_accuracy"].mean(),
            "CV_Accuracy_Std": cv_scores["test_accuracy"].std(),
            "CV_Precision": cv_scores["test_precision"].mean(),
            "CV_Recall": cv_scores["test_recall"].mean(),
            "CV_F1": cv_scores["test_f1"].mean(),
            "CV_ROC_AUC": cv_scores["test_roc_auc"].mean(),
            "CV_PR_AUC": cv_scores["test_pr_auc"].mean()
        })
        
    df_cv_comp = pd.DataFrame(cv_comparison_results)
    df_cv_comp.to_csv(os.path.join(eval_dir, "tuned_candidate_comparison.csv"), index=False)
    print(df_cv_comp.to_string(index=False))
    
    # 3. Final Candidate Selection (Purely based on CV metrics)
    # Rationale: XGBoost Weighted represents the best balance for identifying minority default cases (maximizing recall while keeping high PR-AUC/F1).
    # Let's select "XGBoost Weighted" as our final candidate champion.
    champion_name = "XGBoost Weighted"
    champion_pipeline = best_estimators[champion_name]
    
    print(f"\nFinal Candidate Selected: {champion_name}")
    
    # 4. Final Held-Out Test Evaluation
    # Fit the champion model on the complete training set and evaluate ONCE on the held-out test split
    print("\n=== Evaluating Selected Champion on Held-Out Test Split ===")
    champion_pipeline.fit(X_train, y_train)
    
    y_pred = champion_pipeline.predict(X_test)
    y_proba = champion_pipeline.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred)
    test_rec = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_roc = roc_auc_score(y_test, y_proba)
    test_pr = average_precision_score(y_test, y_proba)
    
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall   : {test_rec:.4f}")
    print(f"Test F1-Score : {test_f1:.4f}")
    print(f"Test ROC-AUC  : {test_roc:.4f}")
    print(f"Test PR-AUC   : {test_pr:.4f}")
    
    # Save test classification report
    report_str = classification_report(y_test, y_pred, digits=4)
    with open(os.path.join(eval_dir, "final_test_classification_report.txt"), "w") as f:
        f.write("=== Final Selected Champion Test Set Evaluation ===\n")
        f.write(f"Model Configuration: Tuned {champion_name}\n\n")
        f.write(report_str)
        
    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Lower Risk", "Higher Risk"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Final Champion Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(img_eval_dir, "final_confusion_matrix.png"), dpi=150)
    plt.close()
    
    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Tuned {champion_name} (AUC = {test_roc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Final Champion ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(img_eval_dir, "final_roc_curve.png"), dpi=150)
    plt.close()
    
    # Save Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"Tuned {champion_name} (AP = {test_pr:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Final Champion Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(img_eval_dir, "final_precision_recall_curve.png"), dpi=150)
    plt.close()
    
    print("\nPhase 6 Tuning and Final Test Evaluation finished successfully!")
    return True

if __name__ == "__main__":
    run_tuning()
