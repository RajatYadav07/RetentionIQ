import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.api.main import app
from app.api.database import Base, engine, SessionLocal
from app.api.models import ActionQueue

# Create a clean test database
Base.metadata.create_all(bind=engine)
client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "RetentionIQ"}

def test_get_actions_empty():
    response = client.get("/actions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict_invalid_data():
    # Missing required fields
    invalid_data = {
        "customer_id": "C001"
        # missing all other fields like country, industry, etc.
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422 # Pydantic Validation Error

def test_get_invalid_customer():
    response = client.get("/customer/DOES_NOT_EXIST")
    assert response.status_code in [404, 503] # 404 if data exists but not found, 503 if master dataset missing

def test_action_creation_and_update():
    action_data = {
        "customer_id": "C001",
        "priority_score": 85.5,
        "priority": "Critical",
        "action": "Executive Check-in",
        "reason": "High risk",
        "urgency": "Critical",
        "objective": "Retain"
    }
    
    # Create
    response = client.post("/actions", json=action_data)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "C001"
    assert data["status"] == "New"
    
    action_id = data["id"]
    
    # Update Status
    update_res = client.put(f"/actions/{action_id}/status", json={"status": "In Progress"})
    assert update_res.status_code == 200
    assert update_res.json()["status"] == "In Progress"
    
    # Duplicate active action should fail
    dup_res = client.post("/actions", json=action_data)
    assert dup_res.status_code == 400
    
    # Clean up test DB
    db = SessionLocal()
    db.query(ActionQueue).filter(ActionQueue.id == action_id).delete()
    db.commit()
    db.close()
