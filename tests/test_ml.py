import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.builder import FeatureEngineer, get_preprocessor

def test_feature_engineer_output_columns():
    """Ensure feature engineering generates the expected columns and handles missing data."""
    # Create mock raw data
    raw_data = pd.DataFrame([{
        "annual_revenue": 10000,
        "tenure_months": 12,
        "discount_percent": 10,
        "login_frequency": 0, # Edge case: division by zero
        "weekly_active_days": 3,
        "feature_adoption_rate": 0.5,
        "support_tickets_30d": 1,
        "support_tickets_90d": 5,
        "open_tickets": 1,
        "avg_resolution_hours": 24,
        "escalation_count": 0,
        "contract_type": "Month-to-Month",
        "auto_renew": True,
        "sentiment_trend": np.nan, # Edge case: missing NLP score
        "usage_change_30d": np.nan, # Edge case: missing usage
        "usage_change_90d": np.nan
    }])
    
    # Process
    processed_df = FeatureEngineer.transform(raw_data)
    
    # Assertions for engineered columns
    assert 'customer_value_score' in processed_df.columns
    assert 'sentiment_trend' in processed_df.columns
    assert 'customer_support_health_score' in processed_df.columns
    assert 'days_since_last_activity' in processed_df.columns
    
    # Test division by zero handlers
    assert processed_df['days_since_last_activity'].iloc[0] == 60 # Capped at 60
    assert not np.isnan(processed_df['sentiment_trend'].iloc[0]) # Should be 0
    assert not np.isnan(processed_df['usage_change_30d'].iloc[0]) # Should be 0
    
def test_preprocessor_target_leakage():
    """Ensure ground truth target columns are dropped by the preprocessor if they accidentally sneak in."""
    raw_data = pd.DataFrame([{
        "customer_id": "C001",
        "annual_revenue": 10000,
        "monthly_charges": 833.33,
        "tenure_months": 12,
        "discount_percent": 10,
        "login_frequency": 5,
        "weekly_active_days": 3,
        "monthly_active_users": 10,
        "feature_adoption_rate": 0.5,
        "support_tickets_30d": 1,
        "support_tickets_90d": 5,
        "open_tickets": 1,
        "avg_resolution_hours": 24,
        "escalation_count": 0,
        "contract_type": "Month-to-Month",
        "auto_renew": True,
        "sentiment_trend": 0.5,
        "usage_change_30d": 5,
        "usage_change_90d": 5,
        "country": "USA",
        "industry": "Tech",
        "company_size": "1-10",
        "acquisition_channel": "Paid",
        "plan_type": "Starter",
        "payment_method": "Credit Card",
        "emails_opened": 5,
        "emails_clicked": 1,
        "csat_score": 4.0,
        "nps_score": 8.0,
        "churn_probability_ground_truth": 0.99, # LEAKAGE
        "churn_date": "2024-01-01",             # LEAKAGE
        "churned": 1                            # TARGET
    }])
    
    # Engineer
    engineered = FeatureEngineer.transform(raw_data)
    
    # Fit transform preprocessor
    preprocessor = get_preprocessor()
    final_matrix = preprocessor.fit_transform(engineered)
    
    # Check that leakage columns were dropped
    final_cols = final_matrix.columns.tolist()
    
    for col in final_cols:
        assert 'churn' not in col.lower(), f"Leakage detected: {col} is in the final feature matrix."
        assert 'customer_id' not in col.lower(), f"Customer ID leaked into matrix: {col}"
