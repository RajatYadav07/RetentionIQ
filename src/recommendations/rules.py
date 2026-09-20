import pandas as pd
import numpy as np

def calculate_priority_score(row, churn_prob):
    """
    Calculates a 0-100 Retention Priority Score.
    """
    # 40% Churn Probability
    prob_score = churn_prob * 40.0
    
    # 20% Customer Value (Capped at $50k ARR)
    arr = row.get('annual_revenue', 0)
    value_score = min(arr / 50000.0, 1.0) * 20.0
    
    # 15% Support Severity
    health = row.get('customer_support_health_score', 100)
    support_score = ((100.0 - health) / 100.0) * 15.0
    
    # 10% Engagement Decline
    usage_change = row.get('usage_change_30d', 0)
    engagement_score = (abs(min(usage_change, 0)) / 50.0)
    engagement_score = min(engagement_score, 1.0) * 10.0
    
    # 15% Contract Urgency
    days_left = row.get('contract_days_remaining', 90)
    contract_score = max(0.0, (90.0 - days_left) / 90.0) * 15.0
    
    total_score = prob_score + value_score + support_score + engagement_score + contract_score
    return min(total_score, 100.0)

def generate_recommendations(df_customer, churn_probability):
    """
    Generates actionable recommendations and priority scores based on customer state.
    """
    row = df_customer.iloc[0] if isinstance(df_customer, pd.DataFrame) else df_customer
    
    # Calculate Priority Score
    priority_score = calculate_priority_score(row, churn_probability)
    
    # Default State
    priority = "Low"
    action = "No immediate action required"
    reason = "Risk and value metrics are stable"
    urgency = "Low"
    objective = "Maintain current relationship"
    
    # Determine Priority Buckets
    if priority_score >= 70:
        priority = "Critical"
    elif priority_score >= 50:
        priority = "High"
    elif priority_score >= 30:
        priority = "Medium"
        
    # Rule Engine
    if priority in ["Critical", "High"]:
        arr = row.get('annual_revenue', 0)
        topic = row.get('dominant_support_topic', 'None')
        sentiment = row.get('sentiment_trend', 0)
        usage_drop = row.get('usage_change_30d', 0)
        days_left = row.get('contract_days_remaining', 90)
        discount = row.get('discount_percent', 0)
        
        # 1. High Churn + High Revenue
        if arr > 20000 and churn_probability > 0.6:
            action = "Executive/Customer Success Intervention"
            reason = f"Critical ARR at risk ({arr:,.0f}) with high churn probability."
            urgency = "Critical"
            objective = "Secure enterprise account via white-glove check-in."
            
        # 2. High Churn + Technical Issues
        elif topic == 'Technical' and row.get('open_tickets', 0) > 0:
            action = "Technical Escalation"
            reason = "Customer is struggling with open technical issues."
            urgency = "High"
            objective = "Resolve blockers to restore platform trust."
            
        # 3. High Churn + Negative Support Sentiment
        elif sentiment < -0.2:
            action = "Priority Support Intervention"
            reason = "Highly negative sentiment detected in recent support interactions."
            urgency = "High"
            objective = "De-escalate frustration and improve CSAT."
            
        # 4. High Churn + Declining Usage
        elif usage_drop < -15:
            action = "Product Adoption Campaign"
            reason = f"Usage velocity dropped by {abs(usage_drop):.1f}% in the last 30 days."
            urgency = "High"
            objective = "Re-engage users with core features."
            
        # 5. High Churn + Contract Ending Soon
        elif days_left <= 30:
            action = "Renewal Outreach"
            reason = f"Contract expires in {days_left:.0f} days with high churn risk."
            urgency = "High"
            objective = "Proactively negotiate renewal terms."
            
        # 6. High Churn + Price Sensitivity
        elif discount == 0:
            action = "Pricing/Plan Review"
            reason = "No active discount on an at-risk account."
            urgency = "Medium"
            objective = "Offer targeted retention discount."
            
        else:
            action = "General CSM Check-in"
            reason = "Elevated risk score requires manual review."
            urgency = "Medium"
            objective = "Identify hidden churn drivers."
            
    elif priority == "Medium":
        if row.get('usage_change_90d', 0) < -10:
            action = "Automated Re-engagement Email"
            reason = "Gradual usage decline over 90 days."
            urgency = "Medium"
            objective = "Drive login frequency."
        else:
            action = "Monitor Account"
            reason = "Medium priority score without critical drivers."
            urgency = "Low"
            objective = "Ensure stability."

    # Map older code that expected 'risk_level' and 'recommended_action'
    # Keeping backwards compatibility for the existing API/Dashboard initially
    return {
        "risk_level": priority, # Map priority to risk_level for backwards compat
        "priority_score": float(priority_score),
        "priority": priority,
        "action": action,
        "recommended_action": action, # Map action for backwards compat
        "reason": reason,
        "urgency": urgency,
        "expected_objective": objective
    }
