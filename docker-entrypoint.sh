#!/bin/bash
set -e

echo "Starting RetentionIQ Docker Initialization..."

# Check if model files exist
if [ ! -f "models/model_pipeline.joblib" ]; then
    echo "Model pipeline not found. Automatically triggering data generation and training..."
    python scripts/train_model.py
else
    echo "Model pipeline exists. Skipping training."
fi

# Hand over execution to the main command (either uvicorn or streamlit passed in via docker-compose)
exec "$@"
