import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config import MODELS_DIR, TARGET_COL, RANDOM_STATE, TEST_SIZE
from src.features.builder import FeatureEngineer, get_preprocessor

def evaluate_model(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob)
    }

def optimize_threshold(y_true, y_prob):
    """Finds the threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Calculate F1 for all thresholds
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    # thresholds array has 1 less element than precisions/recalls
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(best_threshold)

def train_models(df):
    print("Preparing data and strict Train/Test splits...")
    
    # 1. Strict Split Before Any Preprocessing (No Leakage)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 2. Build Unified Baseline Pipeline
    print("Training Baseline Pipeline (Logistic Regression)...")
    baseline_pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer),
        ('preprocessor', get_preprocessor()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    
    baseline_pipeline.fit(X_train, y_train)
    lr_prob = baseline_pipeline.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_model(y_test, lr_prob)
    
    # 3. Build Unified Advanced Pipeline with Calibration
    print("Training Advanced Pipeline (Calibrated XGBoost)...")
    
    xgb_base = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Use cv='prefit' inside the pipeline would be complex.
    # We calibrate XGBoost via cross-validation over the resampled data.
    # To keep pipeline simple and avoid nested cross-val resampling issues:
    # We will wrap the classifier in CalibratedClassifierCV with cv=3
    calibrated_xgb = CalibratedClassifierCV(estimator=xgb_base, method='sigmoid', cv=3)
    
    xgb_pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer),
        ('preprocessor', get_preprocessor()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', calibrated_xgb)
    ])
    
    xgb_pipeline.fit(X_train, y_train)
    
    print("Optimizing Decision Threshold...")
    xgb_prob = xgb_pipeline.predict_proba(X_test)[:, 1]
    best_threshold = optimize_threshold(y_test, xgb_prob)
    
    print(f"Optimal Threshold (Max F1): {best_threshold:.3f}")
    xgb_metrics = evaluate_model(y_test, xgb_prob, threshold=best_threshold)
    
    # Business Impact Calculation
    predicted_churners_idx = xgb_prob >= best_threshold
    rev_at_risk = float(X_test.loc[predicted_churners_idx, 'annual_revenue'].sum())
    
    xgb_metrics['optimal_threshold'] = best_threshold
    xgb_metrics['test_revenue_at_risk_flagged'] = rev_at_risk
    xgb_metrics['test_customers_flagged'] = int(predicted_churners_idx.sum())
    
    print(f"Baseline Metrics (T=0.5): {lr_metrics}")
    print(f"XGBoost Metrics (T={best_threshold:.3f}): {xgb_metrics}")
    
    # Save the unified pipeline
    joblib.dump(xgb_pipeline, MODELS_DIR / "model_pipeline.joblib")
    
    # Extract feature names from preprocessor for metadata
    # Fit the preprocessor manually just to get feature names for the metadata
    dummy_feat = FeatureEngineer.transform(X_train.head(10))
    dummy_prep = get_preprocessor().fit(dummy_feat)
    feature_names = dummy_prep.get_feature_names_out().tolist()
    
    # Save robust metrics
    metrics = {
        "model_version": "1.1.0",
        "training_timestamp": datetime.utcnow().isoformat(),
        "baseline_lr": lr_metrics,
        "xgboost": xgb_metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "positive_class_rate": float(y_train.mean()),
        "features": feature_names
    }
    
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Unified Pipeline and metadata saved successfully.")
    return xgb_pipeline

if __name__ == "__main__":
    from configs.config import PROCESSED_DATA_DIR
    df = pd.read_csv(PROCESSED_DATA_DIR / "master_data_segmented.csv")
    train_models(df)
