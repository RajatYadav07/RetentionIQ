from pydantic import BaseModel, Field
from typing import List, Optional

class CustomerData(BaseModel):
    customer_id: str = Field(..., description="Unique customer ID")
    country: str
    industry: str
    company_size: str
    acquisition_channel: str
    plan_type: str
    contract_type: str
    payment_method: str
    tenure_months: int
    monthly_charges: float
    annual_revenue: float
    discount_percent: float
    auto_renew: bool
    login_frequency: float
    weekly_active_days: float
    monthly_active_users: float
    feature_adoption_rate: float
    usage_change_30d: float
    usage_change_90d: float
    support_tickets_30d: int
    support_tickets_90d: int
    open_tickets: int
    avg_resolution_hours: float
    escalation_count: int
    emails_opened: int
    emails_clicked: int
    csat_score: float
    nps_score: float

class CustomerDataBatch(BaseModel):
    customers: List[CustomerData]

class FeatureImportance(BaseModel):
    feature: str
    impact: float

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str
    recommended_action: str
    reason: str
    urgency: str
    top_drivers: List[FeatureImportance]
    top_protectors: List[FeatureImportance]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class MetricsResponse(BaseModel):
    baseline_lr: dict
    xgboost: dict
    train_size: int
    test_size: int
    positive_class_rate: float
    
class ActionBase(BaseModel):
    customer_id: str
    priority_score: float
    priority: str
    action: str
    reason: str
    urgency: str
    objective: str
    status: str = "New"

class ActionCreate(ActionBase):
    pass

class ActionUpdate(BaseModel):
    status: str

class ActionSchema(ActionBase):
    id: int
    
    class Config:
        from_attributes = True
