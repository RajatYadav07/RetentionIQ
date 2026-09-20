import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = BASE_DIR / "retentioniq.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# ML Settings
TARGET_COL = "churned"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Categorical Features
CATEGORICAL_FEATURES = [
    "country",
    "industry",
    "company_size",
    "acquisition_channel",
    "plan_type",
    "contract_type",
    "payment_method"
]
