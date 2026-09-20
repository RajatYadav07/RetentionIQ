import pandas as pd
import numpy as np
import re
import joblib
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config import MODELS_DIR

def clean_text(text):
    """Normalizes and cleans ticket descriptions."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation & special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class SupportNLPPipeline:
    def __init__(self):
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            
        self.sia = SentimentIntensityAnalyzer()
        self.classifier_pipeline = None
        self.label_encoder = None
        
    def train(self, df_tickets):
        """
        Trains the TF-IDF vectorizer and Issue Classifier on historical tickets.
        Saves the trained models.
        """
        print("Training NLP Support Pipeline...")
        # Clean texts
        df_tickets['cleaned_text'] = df_tickets['description'].apply(clean_text)
        
        X = df_tickets['cleaned_text']
        y = df_tickets['issue_category']
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.classifier_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        
        self.classifier_pipeline.fit(X, y_encoded)
        
        # Save models
        joblib.dump(self.classifier_pipeline, MODELS_DIR / "nlp_classifier.joblib")
        joblib.dump(self.label_encoder, MODELS_DIR / "nlp_label_encoder.joblib")
        print("NLP Support Pipeline trained and saved.")

    def load_models(self):
        """Loads trained NLP models from disk."""
        self.classifier_pipeline = joblib.load(MODELS_DIR / "nlp_classifier.joblib")
        self.label_encoder = joblib.load(MODELS_DIR / "nlp_label_encoder.joblib")
        
    def process_tickets(self, df_tickets):
        """
        Processes new tickets: cleans text, computes sentiment, predicts issue category.
        Returns the enhanced dataframe.
        """
        if self.classifier_pipeline is None or self.label_encoder is None:
            self.load_models()
            
        df_out = df_tickets.copy()
        
        # 1. Clean Text
        df_out['cleaned_text'] = df_out['description'].apply(clean_text)
        
        # 2. Sentiment Scoring (Normalized -1 to 1)
        df_out['nlp_sentiment_score'] = df_out['cleaned_text'].apply(
            lambda x: self.sia.polarity_scores(x)['compound'] if x else 0.0
        )
        
        # 3. Issue Classification
        predicted_encoded = self.classifier_pipeline.predict(df_out['cleaned_text'])
        df_out['predicted_issue_category'] = self.label_encoder.inverse_transform(predicted_encoded)
        
        return df_out

if __name__ == "__main__":
    from configs.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    df = pd.read_csv(RAW_DATA_DIR / "nlp_tickets.csv")
    nlp_pipeline = SupportNLPPipeline()
    nlp_pipeline.train(df)
    
    df_processed = nlp_pipeline.process_tickets(df)
    df_processed.to_csv(PROCESSED_DATA_DIR / "nlp_tickets_processed.csv", index=False)
    print("NLP processing complete. Sample outputs:")
    print(df_processed[['description', 'predicted_issue_category', 'nlp_sentiment_score']].head())
