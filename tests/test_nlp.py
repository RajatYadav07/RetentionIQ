import pandas as pd
import numpy as np
import pytest
from src.nlp.processor import clean_text, SupportNLPPipeline

def test_clean_text():
    # Normal case
    assert clean_text("Hello World!") == "hello world"
    # Null cases
    assert clean_text(None) == ""
    assert clean_text(np.nan) == ""
    # Heavy punctuation
    assert clean_text("System is down!!! Please fix ASAP... @admin") == "system is down please fix asap admin"
    # Extra spaces
    assert clean_text("  Too   many   spaces  ") == "too many spaces"

def test_nlp_pipeline():
    # Dummy data
    df = pd.DataFrame({
        "ticket_id": ["1", "2", "3", "4"],
        "customer_id": ["C1", "C2", "C3", "C4"],
        "description": [
            "I am very happy with this product. It works perfectly.",
            "Terrible! The system is broken and I demand a refund.",
            "How do I reset my password?",
            None
        ],
        "issue_category": ["Account", "Billing", "Technical", "Account"]
    })
    
    pipeline = SupportNLPPipeline()
    pipeline.train(df)
    
    df_processed = pipeline.process_tickets(df)
    
    # Assert dimensions
    assert len(df_processed) == 4
    
    # Assert new columns exist
    assert "cleaned_text" in df_processed.columns
    assert "nlp_sentiment_score" in df_processed.columns
    assert "predicted_issue_category" in df_processed.columns
    
    # Assert Sentiment Bounds
    assert df_processed["nlp_sentiment_score"].max() <= 1.0
    assert df_processed["nlp_sentiment_score"].min() >= -1.0
    
    # Check specific sentiments
    # Ticket 1 should be positive
    assert df_processed.loc[0, "nlp_sentiment_score"] > 0
    # Ticket 2 should be negative
    assert df_processed.loc[1, "nlp_sentiment_score"] < 0
    # Ticket 4 (None) should have sentiment 0.0
    assert df_processed.loc[3, "nlp_sentiment_score"] == 0.0
