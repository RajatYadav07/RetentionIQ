import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_and_clean_data():
    """Loads raw data, joins it, and cleans it."""
    print("Loading raw data...")
    customers = pd.read_csv(RAW_DATA_DIR / "customers.csv")
    subs = pd.read_csv(RAW_DATA_DIR / "subscriptions.csv")
    usage = pd.read_csv(RAW_DATA_DIR / "product_usage.csv")
    tickets = pd.read_csv(RAW_DATA_DIR / "support_tickets.csv")
    interactions = pd.read_csv(RAW_DATA_DIR / "customer_interactions.csv")
    labels = pd.read_csv(RAW_DATA_DIR / "churn_labels.csv")
    nlp_tickets = pd.read_csv(RAW_DATA_DIR / "nlp_tickets.csv")
    
    print("Running NLP Pipeline on Tickets...")
    from src.nlp.processor import SupportNLPPipeline
    nlp_pipeline = SupportNLPPipeline()
    nlp_pipeline.train(nlp_tickets) # Train on all tickets
    nlp_processed = nlp_pipeline.process_tickets(nlp_tickets)
    
    print("Aggregating NLP data for Customer Level...")
    # Aggregate sentiment trend per customer
    sentiment_agg = nlp_processed.groupby('customer_id')['nlp_sentiment_score'].mean().reset_index()
    sentiment_agg.rename(columns={'nlp_sentiment_score': 'sentiment_trend'}, inplace=True)
    
    # Get most frequent predicted issue category per customer
    dominant_topic = nlp_processed.groupby('customer_id')['predicted_issue_category'].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    dominant_topic.rename(columns={'predicted_issue_category': 'dominant_support_topic'}, inplace=True)
    
    print("Merging datasets...")
    df = customers.merge(subs, on="customer_id", how="left")
    df = df.merge(usage, on="customer_id", how="left")
    df = df.merge(tickets, on="customer_id", how="left")
    df = df.merge(interactions, on="customer_id", how="left")
    df = df.merge(sentiment_agg, on="customer_id", how="left")
    df = df.merge(dominant_topic, on="customer_id", how="left")
    df = df.merge(labels, on="customer_id", how="left")
    
    print("Handling missing values...")
    # Industry might have missing values -> fill with 'Unknown'
    df["industry"] = df["industry"].fillna("Unknown")
    
    # CSAT/NPS might have missing values -> fill with median
    df["csat_score"] = df["csat_score"].fillna(df["csat_score"].median())
    df["nps_score"] = df["nps_score"].fillna(df["nps_score"].median())
    
    # Fill missing sentiment trend with 0 (neutral) for users with no tickets
    df["sentiment_trend"] = df["sentiment_trend"].fillna(0)
    
    # Fill missing dominant topic with 'None' for users with no tickets
    df["dominant_support_topic"] = df["dominant_support_topic"].fillna("None")
    
    print("Saving master dataset...")
    df.to_csv(PROCESSED_DATA_DIR / "master_data.csv", index=False)
    
    return df

if __name__ == "__main__":
    load_and_clean_data()
