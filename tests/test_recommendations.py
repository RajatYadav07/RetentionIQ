import pandas as pd
import pytest
from src.recommendations.rules import calculate_priority_score, generate_recommendations

def test_priority_score_bounds():
    # Max possible risk
    row_high = pd.Series({
        'annual_revenue': 100000, # Maxes out the 50k cap
        'customer_support_health_score': 0, # Worst health
        'usage_change_30d': -100, # Max engagement decline
        'contract_days_remaining': 0 # Highest contract urgency
    })
    
    score = calculate_priority_score(row_high, churn_prob=1.0)
    assert score <= 100.0
    assert score > 90.0 # Should be very high
    
    # Min possible risk
    row_low = pd.Series({
        'annual_revenue': 0,
        'customer_support_health_score': 100,
        'usage_change_30d': 50,
        'contract_days_remaining': 365
    })
    
    score_low = calculate_priority_score(row_low, churn_prob=0.0)
    assert score_low == 0.0

def test_recommendation_rules():
    # Rule 1: High Churn + High Revenue -> Executive Intervention
    row_exec = pd.DataFrame([{
        'annual_revenue': 100000,
        'dominant_support_topic': 'None',
        'sentiment_trend': 0,
        'usage_change_30d': 0,
        'contract_days_remaining': 365,
        'discount_percent': 10
    }])
    
    rec_exec = generate_recommendations(row_exec, 0.9)
    assert rec_exec['priority'] in ["Critical", "High"]
    assert rec_exec['action'] == "Executive/Customer Success Intervention"
    
    # Rule 2: High Churn + Tech Issues -> Technical Escalation
    row_tech = pd.DataFrame([{
        'annual_revenue': 10000,
        'dominant_support_topic': 'Technical',
        'open_tickets': 1,
        'sentiment_trend': 0,
        'usage_change_30d': 0,
        'contract_days_remaining': 0,
        'discount_percent': 10
    }])
    
    rec_tech = generate_recommendations(row_tech, 0.9)
    assert rec_tech['action'] == "Technical Escalation"
    
    # Rule 3: High Churn + Negative Sentiment -> Priority Support
    row_sent = pd.DataFrame([{
        'annual_revenue': 10000,
        'dominant_support_topic': 'Billing',
        'open_tickets': 0,
        'sentiment_trend': -0.5,
        'usage_change_30d': 0,
        'contract_days_remaining': 0,
        'discount_percent': 10
    }])
    
    rec_sent = generate_recommendations(row_sent, 0.9)
    assert rec_sent['action'] == "Priority Support Intervention"
