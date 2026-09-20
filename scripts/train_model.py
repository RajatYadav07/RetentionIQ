import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.clean import load_and_clean_data
from src.segmentation.cluster import train_segmentation
from src.models.train import train_models
from scripts.generate_data import main as generate_data
import pandas as pd
from configs.config import PROCESSED_DATA_DIR

def run_pipeline():
    print("=== Starting ML Pipeline ===")
    
    # 0. Generate Data
    generate_data()
    
    # 1. Clean Data
    df = load_and_clean_data()
    
    # 2. Segment Data
    df = train_segmentation(df)
    
    # Save segmented data
    df.to_csv(PROCESSED_DATA_DIR / "master_data_segmented.csv", index=False)
    
    # 3. Train Models
    train_models(df)
    
    print("=== ML Pipeline Complete ===")

if __name__ == "__main__":
    run_pipeline()
