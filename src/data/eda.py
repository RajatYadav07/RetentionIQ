import pandas as pd
import numpy as np

class RetentionEDA:
    """Reusable analytical workflow for exploring churn data."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def analyze_churn_by_segment(self):
        """Which customer segments churn most?"""
        stats = self.df.groupby('segment_name').agg(
            total_customers=('customer_id', 'count'),
            churn_rate=('churned', 'mean'),
            avg_arr=('annual_revenue', 'mean'),
            total_arr_at_risk=('annual_revenue', lambda x: x[self.df.loc[x.index, 'churned'] == 1].sum())
        ).sort_values('churn_rate', ascending=False)
        return stats
        
    def analyze_usage_decline(self):
        """Does declining product usage predict churn?"""
        self.df['usage_trend'] = np.where(self.df['usage_change_30d'] < 0, 'Declining', 'Stable/Growing')
        stats = self.df.groupby('usage_trend').agg(
            churn_rate=('churned', 'mean'),
            avg_login_freq=('login_frequency', 'mean')
        )
        return stats

    def analyze_support_burden(self):
        """Does support burden correlate with churn?"""
        # Bin support tickets
        self.df['ticket_bin'] = pd.cut(self.df['support_tickets_30d'], bins=[-1, 0, 2, 5, 100], labels=['0', '1-2', '3-5', '5+'])
        stats = self.df.groupby('ticket_bin', observed=False).agg(
            churn_rate=('churned', 'mean'),
            avg_open_tickets=('open_tickets', 'mean'),
            avg_escalations=('escalation_count', 'mean')
        )
        return stats

    def analyze_contract_impact(self):
        """Does contract type influence churn?"""
        stats = self.df.groupby('contract_type').agg(
            churn_rate=('churned', 'mean'),
            customer_count=('customer_id', 'count')
        ).sort_values('churn_rate', ascending=False)
        return stats
        
    def analyze_tenure_effect(self):
        """How does tenure affect churn?"""
        self.df['tenure_bin'] = pd.cut(self.df['tenure_months'], bins=[0, 6, 12, 24, 60], labels=['0-6m', '6-12m', '1-2y', '2y+'])
        stats = self.df.groupby('tenure_bin', observed=False).agg(
            churn_rate=('churned', 'mean'),
            customer_count=('customer_id', 'count')
        )
        return stats

    def analyze_acquisition_channels(self):
        """Which acquisition channels produce higher-risk customers?"""
        stats = self.df.groupby('acquisition_channel').agg(
            churn_rate=('churned', 'mean'),
            avg_arr=('annual_revenue', 'mean')
        ).sort_values('churn_rate', ascending=False)
        return stats
        
    def analyze_sentiment_impact(self):
        """Does negative support sentiment precede churn?"""
        if 'sentiment_trend' not in self.df.columns:
            return None
        self.df['sentiment_bin'] = pd.cut(self.df['sentiment_trend'], bins=[-1.1, -0.1, 0.1, 1.1], labels=['Negative', 'Neutral', 'Positive'])
        stats = self.df.groupby('sentiment_bin', observed=False).agg(
            churn_rate=('churned', 'mean'),
            customer_count=('customer_id', 'count')
        )
        return stats

if __name__ == "__main__":
    # Example usage:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from configs.config import PROCESSED_DATA_DIR
    df = pd.read_csv(PROCESSED_DATA_DIR / "master_data_segmented.csv")
    eda = RetentionEDA(df)
    print("=== Churn by Segment ===")
    print(eda.analyze_churn_by_segment())
    print("\n=== Usage Decline Impact ===")
    print(eda.analyze_usage_decline())
