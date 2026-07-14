import os
import json
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss, log_loss, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.preprocessing import get_preprocessor, TARGET_COLUMN

def run_calibration_and_thresholds():
    cleaned_path = os.path.join("data", "processed", "credit_risk_cleaned.csv")
    eval_dir = os.path.join("models", "evaluation")
    img_eval_dir = os.path.join("static", "images", "model_evaluation")
    models_dir = os.path.join("models")
    
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(img_eval_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(cleaned_path):
        print(f"Error: Cleaned dataset not found at {cleaned_path}")
        return False
        
    df = pd.read_csv(cleaned_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Split: Train/Test split (80/20 stratified, seed 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Calculate pos weight from train only
    scale_weight = float(sum(y_train == 0)) / float(sum(y_train == 1))
    
    # Recreate the best tuned model from Phase 6
    best_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.2,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 5,
        "scale_pos_weight": scale_weight,
        "random_state": 42,
        "eval_metric": "logloss"
    }
    
    xgb_base = xgb.XGBClassifier(**best_params)
    
    base_pipeline = Pipeline(steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", xgb_base)
    ])
    
    # 1. Calibration Method Evaluation (using 80/20 split of training data)
    print("=== Step 1: Calibration Method Evaluation ===")
    X_sub_train, X_sub_val, y_sub_train, y_sub_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    
    # Models to compare
    # Uncalibrated
    model_uncal = Pipeline(steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", xgb.XGBClassifier(**best_params))
    ])
    model_uncal.fit(X_sub_train, y_sub_train)
    
    # Sigmoid / Platt
    model_sig = CalibratedClassifierCV(estimator=model_uncal, method="sigmoid", cv=5)
    model_sig.fit(X_sub_train, y_sub_train)
    
    # Isotonic
    model_iso = CalibratedClassifierCV(estimator=model_uncal, method="isotonic", cv=5)
    model_iso.fit(X_sub_train, y_sub_train)
    
    calibration_models = {
        "Uncalibrated": model_uncal,
        "Sigmoid (Platt)": model_sig,
        "Isotonic": model_iso
    }
    
    calibration_results = []
    
    # Plot Calibration Reliability curves
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    
    for name, model in calibration_models.items():
        y_proba = model.predict_proba(X_sub_val)[:, 1]
            
        brier = brier_score_loss(y_sub_val, y_proba)
        loss = log_loss(y_sub_val, y_proba)
        roc = roc_auc_score(y_sub_val, y_proba)
        pr = average_precision_score(y_sub_val, y_proba)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(y_sub_val, y_proba, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{name} (Brier = {brier:.4f})")
        
        calibration_results.append({
            "Method": name,
            "Brier_Score": brier,
            "Log_Loss": loss,
            "ROC_AUC": roc,
            "PR_AUC": pr
        })
        
    plt.ylabel("Fraction of Positives")
    plt.xlabel("Mean Predicted Probability")
    plt.title("Calibration Reliability Diagrams (Validation Split)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(img_eval_dir, "calibration_reliability.png"), dpi=200)
    plt.close()
    
    df_cal = pd.DataFrame(calibration_results)
    df_cal.to_csv(os.path.join(eval_dir, "calibration_comparison.csv"), index=False)
    print("\nCalibration Comparison Results:")
    print(df_cal.to_string(index=False))
    print("")
    
    # Select best calibrator: Sigmoid Platt vs Isotonic (based on Brier score minimization)
    best_row = df_cal.loc[df_cal["Method"] != "Uncalibrated"].sort_values("Brier_Score").iloc[0]
    best_method = "sigmoid" if "Sigmoid" in best_row["Method"] else "isotonic"
    print(f"Selected Calibration Method based on Brier Score: {best_row['Method']} ({best_method})")
    print("")
    
    # 2. Decision Threshold Analysis (using out-of-fold train split predictions)
    print("=== Step 2: Decision Threshold Analysis ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X_train))
    
    # Generate OOF probabilities
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        fold_uncal = Pipeline(steps=[
            ("preprocessor", get_preprocessor()),
            ("classifier", xgb.XGBClassifier(**best_params))
        ])
        
        fold_cal = CalibratedClassifierCV(estimator=fold_uncal, method=best_method, cv=5)
        fold_cal.fit(X_fold_train, y_fold_train)
        
        oof_probs[val_idx] = fold_cal.predict_proba(X_fold_val)[:, 1]
        
    # Analyze metrics across a threshold range [0.01, 0.99]
    thresholds = np.linspace(0.01, 0.99, 99)
    threshold_records = []
    
    for t in thresholds:
        y_pred = (oof_probs >= t).astype(int)
        acc = accuracy_score(y_train, y_pred)
        precision_val = precision_score(y_train, y_pred, zero_division=0)
        rec = recall_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)
        
        cm = confusion_matrix(y_train, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        pred_high_risk_rate = sum(y_pred == 1) / len(y_pred)
        
        threshold_records.append({
            "Threshold": t,
            "Accuracy": acc,
            "Precision": precision_val,
            "Recall": rec,
            "F1_Score": f1,
            "False_Positive_Rate": fpr,
            "False_Negative_Rate": fnr,
            "Predicted_Higher_Risk_Rate": pred_high_risk_rate
        })
        
    df_thresh = pd.DataFrame(threshold_records)
    df_thresh.to_csv(os.path.join(eval_dir, "threshold_analysis.csv"), index=False)
    
    # Plot precision-recall vs threshold trade-offs
    plt.figure(figsize=(8, 5))
    plt.plot(df_thresh["Threshold"], df_thresh["Precision"], label="Precision", color="#2A9D8F", linewidth=2)
    plt.plot(df_thresh["Threshold"], df_thresh["Recall"], label="Recall", color="#E76F51", linewidth=2)
    plt.plot(df_thresh["Threshold"], df_thresh["F1_Score"], label="F1-Score", color="#F4A261", linestyle="--")
    plt.axvline(0.50, color="gray", linestyle=":", label="Default Threshold (0.50)")
    plt.title("Precision-Recall Trade-offs across Thresholds")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_eval_dir, "threshold_tradeoffs.png"), dpi=200)
    plt.close()
    
    # 3. Model Classification Threshold Selection & Risk Category Design
    # Selection: We keep classification threshold at 0.50 to yield balanced metrics.
    # However, we define user-facing risk categories based on default probability:
    # Low Risk: P(default) < 0.12 (Low observed default rate, high safety profile)
    # Medium Risk: 0.12 <= P(default) < 0.35 (Conditional status, requires review)
    # High Risk: P(default) >= 0.35 (High default probability)
    selected_threshold = 0.50
    low_boundary = 0.12
    high_boundary = 0.35
    
    print("=== Step 3: Risk Classification & Boundaries ===")
    print(f"Model Classification Decision Threshold: {selected_threshold:.2f}")
    print(f"User-Facing Risk Boundaries:")
    print(f"  - Low Risk Category   : Probability < {low_boundary:.2f}")
    print(f"  - Medium Risk Category: {low_boundary:.2f} <= Probability < {high_boundary:.2f}")
    print(f"  - High Risk Category  : Probability >= {high_boundary:.2f}")
    print("")
    
    # 4. Final Model Training (Fit Calibrated classifier on entire Train Set)
    print("=== Step 4: Final Model Persistence ===")
    final_uncal = Pipeline(steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", xgb.XGBClassifier(**best_params))
    ])
    
    final_calibrated_model = CalibratedClassifierCV(estimator=final_uncal, method=best_method, cv=5)
    final_calibrated_model.fit(X_train, y_train)
    
    # Save the calibrated model
    model_artifact_path = os.path.join("models", "credit_risk_model.joblib")
    joblib.dump(final_calibrated_model, model_artifact_path)
    print(f"Production model artifact successfully saved to: {model_artifact_path}")
    
    # 5. Final Held-Out Test Evaluation (Run ONCE)
    print("\n=== Step 5: Final Generalization Evaluation on Held-Out Test Split ===")
    y_test_pred = final_calibrated_model.predict(X_test)
    y_test_proba = final_calibrated_model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc = roc_auc_score(y_test, y_test_proba)
    test_pr = average_precision_score(y_test, y_test_proba)
    test_brier = brier_score_loss(y_test, y_test_proba)
    test_loss = log_loss(y_test, y_test_proba)
    
    print(f"FINAL GENERALIZATION RESULTS (Test Set):")
    print(f"  - Accuracy: {test_acc:.4f}")
    print(f"  - Precision: {test_prec:.4f}")
    print(f"  - Recall   : {test_rec:.4f}")
    print(f"  - F1-Score : {test_f1:.4f}")
    print(f"  - ROC-AUC  : {test_roc:.4f}")
    print(f"  - PR-AUC   : {test_pr:.4f}")
    print(f"  - Brier Score: {test_brier:.4f}")
    print(f"  - Log Loss   : {test_loss:.4f}")
    print("")
    
    # Save final curves on Test set
    # Final Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Lower Risk", "Higher Risk"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Final Calibrated Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(img_eval_dir, "final_confusion_matrix.png"), dpi=200)
    plt.close()
    
    # Final ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Final Calibrated Model (AUC = {test_roc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Final Calibrated Model ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(img_eval_dir, "final_roc_curve.png"), dpi=200)
    plt.close()
    
    # Final PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"Final Calibrated Model (AP = {test_pr:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Final Calibrated Model Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(img_eval_dir, "final_precision_recall_curve.png"), dpi=200)
    plt.close()
    
    # Save test classification report
    report_str = classification_report(y_test, y_test_pred, digits=4)
    with open(os.path.join(eval_dir, "final_calibrated_classification_report.txt"), "w") as f:
        f.write("=== Final Calibrated Selected System Test Evaluation ===\n")
        f.write(report_str)
        
    # Get feature names from fitted ColumnTransformer within CalibratedClassifierCV
    fitted_pipeline = final_calibrated_model.calibrated_classifiers_[0].estimator
    fitted_preprocessor = fitted_pipeline.named_steps["preprocessor"]
    from src.preprocessing import get_feature_names
    feature_names = get_feature_names(fitted_preprocessor)
    
    # 6. Save Metadata JSON
    metadata = {
        "project_name": "CrediSense AI: Explainable Credit Approval & Risk Assessment System",
        "model_type": "CalibratedClassifierCV(XGBClassifier)",
        "model_library_versions": {
            "xgboost": xgb.__version__,
            "scikit-learn": "1.7.2",
            "joblib": "1.5.3"
        },
        "training_date": str(datetime.date.today()),
        "feature_names": feature_names,
        "excluded_features": ["loan_grade", "loan_int_rate"],
        "target_name": TARGET_COLUMN,
        "class_mapping": {
            "0": "Lower observed credit/default risk class (historically did not default)",
            "1": "Higher observed credit/default risk class (historically defaulted)"
        },
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.2,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 5,
            "scale_pos_weight": scale_weight
        },
        "scale_pos_weight_explanation": "Imbalance weighting ratio derived from training-set class ratio (Class 0 count / Class 1 count). Adjusts loss function weight of minority defaults class.",
        "calibration_method": best_method,
        "classification_threshold": selected_threshold,
        "risk_category_boundaries": {
            "low_risk_upper_bound": low_boundary,
            "high_risk_lower_bound": high_boundary
        },
        "cv_metrics": {
            "CV_PR_AUC": float(df_cal.loc[df_cal["Method"] == best_row["Method"], "PR_AUC"].iloc[0]),
            "CV_ROC_AUC": float(df_cal.loc[df_cal["Method"] == best_row["Method"], "ROC_AUC"].iloc[0]),
            "CV_Brier_Score": float(best_row["Brier_Score"]),
            "CV_Log_Loss": float(best_row["Log_Loss"])
        },
        "final_held_out_test_metrics": {
            "Accuracy": float(test_acc),
            "Precision": float(test_prec),
            "Recall": float(test_rec),
            "F1_Score": float(test_f1),
            "ROC_AUC": float(test_roc),
            "PR_AUC": float(test_pr),
            "Brier_Score": float(test_brier),
            "Log_Loss": float(test_loss)
        },
        "dataset_size": len(df),
        "random_state": 42
    }
    
    metadata_path = os.path.join("models", "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Production model metadata successfully saved to: {metadata_path}")
    print("\nPhase 7 Calibration, Thresholds, and Persistence completed successfully!")
    return True

if __name__ == "__main__":
    run_calibration_and_thresholds()
