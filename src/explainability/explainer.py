import pandas as pd
import numpy as np
import shap
import joblib
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config import MODELS_DIR

_PIPELINE_CACHE = None
_EXPLAINER_CACHE = None
_FEATURE_NAMES_CACHE = None

def _get_model_artifacts():
    global _PIPELINE_CACHE, _EXPLAINER_CACHE, _FEATURE_NAMES_CACHE
    if _PIPELINE_CACHE is None:
        _PIPELINE_CACHE = joblib.load(MODELS_DIR / "model_pipeline.joblib")
        calibrated_clf = _PIPELINE_CACHE.named_steps['classifier']
        base_xgb = calibrated_clf.calibrated_classifiers_[0].estimator
        _EXPLAINER_CACHE = shap.TreeExplainer(base_xgb)
        try:
            with open(MODELS_DIR / "metrics.json", "r") as f:
                metrics = json.load(f)
                _FEATURE_NAMES_CACHE = metrics.get('features', None)
        except Exception:
            _FEATURE_NAMES_CACHE = None
    return _PIPELINE_CACHE, _EXPLAINER_CACHE, _FEATURE_NAMES_CACHE

def get_customer_explanation(df_customer):
    """
    Given a single customer DataFrame row (unprocessed),
    returns top drivers and protective factors using the unified pipeline.
    """
    pipeline, explainer, feature_names = _get_model_artifacts()
    
    # Process the data up to the classifier
    # We must run it through the feature engineer and preprocessor manually for SHAP
    X_engineered = pipeline.named_steps['feature_engineer'].transform(df_customer)
    X_processed = pipeline.named_steps['preprocessor'].transform(X_engineered)
    
    # Predict Probability using the full calibrated pipeline!
    prob = pipeline.predict_proba(df_customer)[0, 1]
    
    # Get SHAP values using the cached base TreeExplainer
    shap_values = explainer.shap_values(X_processed)
    
    if feature_names is None:
        feature_names = getattr(X_processed, 'columns', [f"feature_{i}" for i in range(X_processed.shape[1])])
    
    impacts = shap_values[0]
    feature_impacts = list(zip(feature_names, impacts))
    
    # Sort by impact
    sorted_impacts = sorted(feature_impacts, key=lambda x: x[1], reverse=True)
    
    top_drivers = [{"feature": f, "impact": float(i)} for f, i in sorted_impacts if i > 0][:5]
    top_protectors = [{"feature": f, "impact": float(i)} for f, i in sorted_impacts[::-1] if i < 0][:5]
    
    return {
        "churn_probability": float(prob),
        "top_drivers": top_drivers,
        "top_protectors": top_protectors
    }

if __name__ == "__main__":
    from configs.config import PROCESSED_DATA_DIR
    df = pd.read_csv(PROCESSED_DATA_DIR / "master_data_segmented.csv")
    sample_customer = df.drop(columns=['churned']).iloc[[0]]
    explanation = get_customer_explanation(sample_customer)
    print(json.dumps(explanation, indent=4))
