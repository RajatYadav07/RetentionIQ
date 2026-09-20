import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def _engineer_columns(X):
    """
    Internal function to engineer features. 
    Expects a pandas DataFrame and returns a pandas DataFrame.
    """
    df_feat = X.copy()
    
    # 1 & 2. usage_change_30d and usage_change_90d (Already exist in raw data, but ensuring they are explicitly handled if missing)
    df_feat['usage_change_30d'] = df_feat['usage_change_30d'].fillna(0)
    df_feat['usage_change_90d'] = df_feat['usage_change_90d'].fillna(0)
    
    # 3. support_ticket_velocity (Ratio of 30d to 90d to see if support requests are accelerating)
    # Using 90d/3 since it represents a 3-month window
    avg_tickets_per_30d_historically = df_feat['support_tickets_90d'] / 3.0
    df_feat['support_ticket_velocity'] = df_feat['support_tickets_30d'] / (avg_tickets_per_30d_historically.replace(0, 1))
    
    # 4. complaint_ratio (Open tickets relative to total recent tickets)
    df_feat['complaint_ratio'] = df_feat['open_tickets'] / (df_feat['support_tickets_30d'].replace(0, 1))
    
    # 5. engagement_score (Weighted combo of login freq, active days, adoption)
    df_feat['engagement_score'] = (
        (df_feat['login_frequency'] / 30.0) * 0.4 +
        (df_feat['weekly_active_days'] / 7.0) * 0.3 +
        (df_feat['feature_adoption_rate']) * 0.3
    )
    
    # 6. customer_value_score (Proxy for CLV: ARR * Tenure * Discount Modifier)
    discount_modifier = (100 - df_feat.get('discount_percent', 0)) / 100.0
    df_feat['customer_value_score'] = df_feat['annual_revenue'] * df_feat['tenure_months'] * discount_modifier
    
    # 7. days_since_last_activity (Inferred: If login frequency is very low, days since last is high)
    # Assume 30 days in a month. If login_frequency is 0, days since last = 30. If 30, days = 1.
    df_feat['days_since_last_activity'] = 30.0 / (df_feat['login_frequency'].replace(0, 0.5))
    df_feat['days_since_last_activity'] = df_feat['days_since_last_activity'].clip(upper=60) # Cap at 60 days
    
    # 8. contract_days_remaining (Inferred from contract type and tenure)
    # M2M = 30 days. Annual = (12 - (tenure % 12)) * 30. 2-Year = (24 - (tenure % 24)) * 30.
    def calc_remaining_days(row):
        if row['contract_type'] == 'Month-to-Month':
            return 30
        elif row['contract_type'] == 'Annual':
            return (12 - (row['tenure_months'] % 12)) * 30
        elif row['contract_type'] == '2-Year':
            return (24 - (row['tenure_months'] % 24)) * 30
        return 30
        
    if 'contract_type' in df_feat.columns:
        df_feat['contract_days_remaining'] = df_feat.apply(calc_remaining_days, axis=1)
    else:
        df_feat['contract_days_remaining'] = 30
        
    # 9. support_severity_score (Severity based on resolution delays and escalations)
    df_feat['support_severity_score'] = df_feat['avg_resolution_hours'] * (df_feat['escalation_count'] + 1)
    
    # 10. sentiment_trend (Aggregated from NLP, ensuring no missing values)
    if 'sentiment_trend' in df_feat.columns:
        df_feat['sentiment_trend'] = df_feat['sentiment_trend'].fillna(0)
    else:
        df_feat['sentiment_trend'] = 0
        
    # 11. customer_support_health_score (0-100, 100 is perfectly healthy)
    # Start with 100 and deduct penalties
    health_score = 100.0
    
    # Penalty 1: Ticket Volume (up to 20 pts)
    # Base expected: 0-1 tickets/mo.
    ticket_penalty = (df_feat['support_tickets_30d'] / 5.0).clip(upper=1) * 20
    
    # Penalty 2: Open Tickets Ratio (up to 20 pts)
    open_penalty = df_feat['complaint_ratio'] * 20
    
    # Penalty 3: Negative Sentiment (up to 30 pts)
    # sentiment_trend ranges from -1 (Negative) to 1 (Positive)
    # We only penalize if it's negative
    sentiment_penalty = np.where(df_feat['sentiment_trend'] < 0, 
                                 np.abs(df_feat['sentiment_trend']) * 30, 
                                 0)
                                 
    # Penalty 4: Severity & Escalations (up to 30 pts)
    # support_severity_score = avg_resolution * (escalations + 1)
    # A terrible score might be 168 (1 week) * 3 escalations = 504
    severity_penalty = (df_feat['support_severity_score'] / 100.0).clip(upper=1) * 30
    
    final_health = health_score - (ticket_penalty + open_penalty + sentiment_penalty + severity_penalty)
    df_feat['customer_support_health_score'] = final_health.clip(lower=0, upper=100)

    # Old features to retain
    df_feat['usage_change_ratio'] = df_feat['usage_change_30d'] / (df_feat['usage_change_90d'].replace(0, 1))
    df_feat['support_ticket_density'] = df_feat['support_tickets_30d'] / df_feat['tenure_months'].replace(0, 1)
    df_feat['is_contract_ending_soon'] = ((df_feat['contract_type'] == 'Month-to-Month') | 
                                          ((df_feat['contract_type'] != 'Month-to-Month') & (df_feat['tenure_months'] % 12 > 10))).astype(int)
    
    # Ensure boolean is int
    if 'auto_renew' in df_feat.columns:
        df_feat['auto_renew'] = df_feat['auto_renew'].astype(int)
        
    return df_feat

# Expose as a scikit-learn transformer
FeatureEngineer = FunctionTransformer(_engineer_columns, validate=False)

# Define column lists for the ColumnTransformer
NUM_COLS = [
    'tenure_months', 'monthly_charges', 'annual_revenue', 'discount_percent',
    'login_frequency', 'weekly_active_days', 'monthly_active_users', 'feature_adoption_rate',
    'usage_change_30d', 'usage_change_90d', 'support_tickets_30d', 'support_tickets_90d',
    'open_tickets', 'avg_resolution_hours', 'escalation_count',
    'emails_opened', 'emails_clicked', 'csat_score', 'nps_score',
    
    # Old engineered
    'usage_change_ratio', 'support_ticket_density', 'is_contract_ending_soon', 'auto_renew',
    
    # New DS engineered
    'support_ticket_velocity', 'complaint_ratio', 'engagement_score', 
    'customer_value_score', 'days_since_last_activity', 'contract_days_remaining', 
    'support_severity_score', 'sentiment_trend', 'customer_support_health_score'
]

CAT_COLS = [
    'country', 'industry', 'company_size', 'acquisition_channel',
    'plan_type', 'contract_type', 'payment_method'
]

def get_preprocessor():
    """
    Returns an unfitted ColumnTransformer for numerical scaling and categorical encoding.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS)
        ],
        remainder='drop' # Drop columns like customer_id, segment_name, etc.
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor
