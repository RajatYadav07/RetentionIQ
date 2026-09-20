import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config import MODELS_DIR, PROCESSED_DATA_DIR

def train_segmentation(df):
    print("Training customer segmentation model...")
    # Select key numerical features for clustering
    features = [
        'tenure_months', 'annual_revenue', 'usage_change_90d', 
        'support_tickets_90d', 'csat_score', 'login_frequency'
    ]
    
    # Fill any remaining NaNs for clustering
    X = df[features].copy().fillna(df[features].median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    joblib.dump(scaler, MODELS_DIR / "cluster_scaler.joblib")
    joblib.dump(kmeans, MODELS_DIR / "kmeans_model.joblib")
    
    df['segment_id'] = kmeans.labels_
    
    # Assign names based on cluster centers (simplified heuristic for demo)
    # 0: High Value At Risk, 1: High Value Loyal, 2: Support Heavy, 3: New & Growing, 4: Low Engagement
    # To do this accurately, we should analyze the centers, but we will assign generic names and let the UI show the stats.
    
    segment_names = {
        0: "Segment A",
        1: "Segment B",
        2: "Segment C",
        3: "Segment D",
        4: "Segment E"
    }
    
    df['segment_name'] = df['segment_id'].map(segment_names)
    
    # A realistic implementation would map these based on actual sorted values (e.g. highest ARR = High Value)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=features)
    
    # Dynamic naming logic
    arr_ranks = centers_df['annual_revenue'].rank()
    support_ranks = centers_df['support_tickets_90d'].rank()
    tenure_ranks = centers_df['tenure_months'].rank()
    usage_drop_ranks = centers_df['usage_change_90d'].rank(ascending=False)
    
    names = {}
    for i in range(5):
        if arr_ranks[i] >= 4 and usage_drop_ranks[i] >= 4:
            names[i] = "High Value At Risk"
        elif arr_ranks[i] >= 4:
            names[i] = "High Value Loyal"
        elif support_ranks[i] >= 4:
            names[i] = "Support Heavy"
        elif tenure_ranks[i] <= 2:
            names[i] = "New & Growing"
        else:
            names[i] = "Low Engagement"
            
    # Handle duplicates in names if any
    used_names = set()
    for k, v in names.items():
        if v in used_names:
            names[k] = f"{v} (Variant)"
        used_names.add(names[k])
        
    df['segment_name'] = df['segment_id'].map(names)
    print("Segment names mapped:")
    print(df['segment_name'].value_counts())
    
    return df

def predict_segments(df):
    scaler = joblib.load(MODELS_DIR / "cluster_scaler.joblib")
    kmeans = joblib.load(MODELS_DIR / "kmeans_model.joblib")
    
    features = [
        'tenure_months', 'annual_revenue', 'usage_change_90d', 
        'support_tickets_90d', 'csat_score', 'login_frequency'
    ]
    X = df[features].copy().fillna(df[features].median())
    X_scaled = scaler.transform(X)
    
    labels = kmeans.predict(X_scaled)
    # In a real app we'd save the name mapping dictionary too, but for inference we'll just return the ID
    return labels

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_DIR / "master_data.csv")
    df = train_segmentation(df)
    df.to_csv(PROCESSED_DATA_DIR / "master_data_segmented.csv", index=False)
