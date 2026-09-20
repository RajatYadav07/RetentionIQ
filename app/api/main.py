from fastapi import FastAPI, HTTPException
import pandas as pd
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from typing import List
from sqlalchemy.orm import Session
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from app.api.schemas import CustomerData, PredictionResponse, BatchPredictionResponse, CustomerDataBatch, MetricsResponse, ActionCreate, ActionUpdate, ActionSchema
from app.api.database import engine, Base, get_db
from app.api.models import ActionQueue
from src.explainability.explainer import get_customer_explanation
from src.recommendations.rules import generate_recommendations
from configs.config import MODELS_DIR, PROCESSED_DATA_DIR
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RetentionIQ_API")

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="RetentionIQ API",
    description="Customer Intelligence & Retention Platform API for predicting churn and managing retention actions.",
    version="1.0.0",
    contact={
        "name": "RetentionIQ Engineering",
    }
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected internal server error occurred.", "details": str(exc)},
    )

@app.get("/health", tags=["System"])
def health_check():
    """System health check endpoint."""
    return {"status": "ok", "service": "RetentionIQ"}

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict_churn(customer: CustomerData):
    """
    Predicts churn probability for a single customer and provides SHAP explanations and recommendations.
    """
    logger.info(f"Received prediction request for customer: {customer.customer_id}")
    try:
        df_customer = pd.DataFrame([customer.model_dump()])
        explanation = get_customer_explanation(df_customer)
        recommendation = generate_recommendations(df_customer, explanation["churn_probability"])
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=explanation["churn_probability"],
            risk_level=recommendation["risk_level"],
            recommended_action=recommendation["recommended_action"],
            reason=recommendation["reason"],
            urgency=recommendation["urgency"],
            top_drivers=explanation["top_drivers"],
            top_protectors=explanation["top_protectors"]
        )
    except FileNotFoundError:
        logger.error("Model files not found.")
        raise HTTPException(status_code=503, detail="Model is currently unavailable. Please ensure training has completed.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed due to invalid data processing: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
def predict_batch(batch: CustomerDataBatch):
    """Predicts churn for a batch of customers."""
    if not batch.customers:
        raise HTTPException(status_code=400, detail="Empty batch provided.")
        
    responses = []
    try:
        for customer in batch.customers:
            df_customer = pd.DataFrame([customer.model_dump()])
            explanation = get_customer_explanation(df_customer)
            recommendation = generate_recommendations(df_customer, explanation["churn_probability"])
            
            res = PredictionResponse(
                customer_id=customer.customer_id,
                churn_probability=explanation["churn_probability"],
                risk_level=recommendation["risk_level"],
                recommended_action=recommendation["recommended_action"],
                reason=recommendation["reason"],
                urgency=recommendation["urgency"],
                top_drivers=explanation["top_drivers"],
                top_protectors=explanation["top_protectors"]
            )
            responses.append(res)
        return BatchPredictionResponse(predictions=responses)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model unavailable.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/customer/{customer_id}", tags=["Customers"])
def get_customer(customer_id: str):
    """Retrieves raw data and predicted metrics for a specific customer."""
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "master_data_segmented.csv")
        cust = df[df['customer_id'] == customer_id]
        if cust.empty:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found.")
        return cust.to_dict(orient="records")[0]
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Master dataset not found. Data pipeline must be executed.")

@app.get("/customers/high-risk", tags=["Customers"])
def get_high_risk_customers():
    """Retrieves all customers with a churn probability > 0.5."""
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "master_data_segmented.csv")
        high_risk = df[df['churn_probability_ground_truth'] > 0.5]
        return high_risk.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Master dataset not found.")

@app.get("/metrics", response_model=MetricsResponse, tags=["Model Info"])
def get_model_metrics():
    """Returns model performance metrics from training."""
    try:
        with open(MODELS_DIR / "metrics.json", "r") as f:
            metrics = json.load(f)
        return MetricsResponse(**metrics)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Metrics file not found. Model might not be trained yet.")
        
@app.get("/model/info", tags=["Model Info"])
def get_model_info():
    """Returns metadata about the active model pipeline."""
    try:
        import joblib
        # Check if pipeline exists
        if not (MODELS_DIR / "model_pipeline.joblib").exists():
            raise FileNotFoundError
        return {
            "status": "Active",
            "type": "XGBoost + CalibratedClassifierCV + imblearn Pipeline",
            "features_expected": 32,
            "version": "1.0.0"
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model pipeline not found.")

# --- Action Queue Endpoints ---

@app.get("/actions", response_model=List[ActionSchema], tags=["Actions"])
def get_actions(status: str = None, db: Session = Depends(get_db)):
    """Returns all queued retention actions, optionally filtered by status."""
    query = db.query(ActionQueue)
    if status and status != "All":
        query = query.filter(ActionQueue.status == status)
    return query.order_by(ActionQueue.priority_score.desc()).all()

@app.post("/actions", response_model=ActionSchema, tags=["Actions"])
def create_action(action: ActionCreate, db: Session = Depends(get_db)):
    """Creates a new retention action in the queue."""
    existing = db.query(ActionQueue).filter(
        ActionQueue.customer_id == action.customer_id,
        ActionQueue.status.in_(["New", "In Progress", "Contacted"])
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="An active action already exists for this customer.")
        
    db_action = ActionQueue(**action.model_dump())
    db.add(db_action)
    db.commit()
    db.refresh(db_action)
    return db_action

@app.put("/actions/{action_id}/status", response_model=ActionSchema, tags=["Actions"])
def update_action_status(action_id: int, update: ActionUpdate, db: Session = Depends(get_db)):
    """Updates the status of an existing action."""
    db_action = db.query(ActionQueue).filter(ActionQueue.id == action_id).first()
    if not db_action:
        raise HTTPException(status_code=404, detail="Action not found")
        
    db_action.status = update.status
    db.commit()
    db.refresh(db_action)
    return db_action
