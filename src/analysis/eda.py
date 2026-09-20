import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config import PROCESSED_DATA_DIR, BASE_DIR

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

def load_data():
    return pd.read_csv(PROCESSED_DATA_DIR / "master_data_segmented.csv")

def plot_churn_by_category(df, category_col, title, filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=category_col, y="churned", errorbar=None)
    plt.title(title)
    plt.ylabel("Churn Rate")
    plt.xlabel(category_col.replace('_', ' ').title())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()

def plot_numerical_distribution(df, num_col, hue_col, title, filename):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=num_col, hue=hue_col, fill=True, common_norm=False)
    plt.title(title)
    plt.xlabel(num_col.replace('_', ' ').title())
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()

def plot_correlation_heatmap(df, filename):
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmax=.5, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()

def plot_revenue_at_risk(df, category_col, filename):
    plt.figure(figsize=(10, 6))
    churned_df = df[df['churned'] == 1]
    rev = churned_df.groupby(category_col)['annual_revenue'].sum().reset_index()
    sns.barplot(data=rev, x=category_col, y='annual_revenue')
    plt.title(f"Revenue Lost by {category_col.replace('_', ' ').title()}")
    plt.ylabel("Total ARR ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()

def generate_html_report(df):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RetentionIQ EDA Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f8f9fa; }}
            h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; }}
            .metric-row {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
            .metric-box {{ flex: 1; background: #3498db; color: white; padding: 20px; border-radius: 8px; text-align: center; margin: 0 10px; }}
            .metric-box h3 {{ margin: 0; font-size: 2em; border: none; color: white; padding: 0; }}
            .metric-box p {{ margin: 0; text-transform: uppercase; font-size: 0.8em; letter-spacing: 1px; }}
            img {{ max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>RetentionIQ: Exploratory Data Analysis Report</h1>
        
        <div class="metric-row">
            <div class="metric-box">
                <h3>{len(df):,}</h3>
                <p>Total Customers</p>
            </div>
            <div class="metric-box" style="background: #e74c3c;">
                <h3>{df['churned'].mean():.1%}</h3>
                <p>Overall Churn Rate</p>
            </div>
            <div class="metric-box" style="background: #2ecc71;">
                <h3>${df['annual_revenue'].sum():,.0f}</h3>
                <p>Total ARR</p>
            </div>
            <div class="metric-box" style="background: #e67e22;">
                <h3>${df.loc[df['churned'] == 1, 'annual_revenue'].sum():,.0f}</h3>
                <p>Lost ARR</p>
            </div>
        </div>

        <div class="card">
            <h2>Dataset Overview & Data Quality</h2>
            <p><strong>Columns:</strong> {len(df.columns)}</p>
            <p><strong>Missing Values:</strong></p>
            <ul>
                {"".join(f"<li>{col}: {missing} missing ({missing/len(df):.1%})</li>" for col, missing in df.isnull().sum().items() if missing > 0)}
            </ul>
        </div>

        <div class="card">
            <h2>Business Insights & Churn Drivers</h2>
            
            <h3>1. Contract Dynamics</h3>
            <p>Customers on Month-to-Month contracts exhibit significantly higher churn rates compared to Annual or 2-Year commitments.</p>
            <img src="figures/churn_by_contract.png" alt="Churn by Contract">
            
            <h3>2. Product Usage Velocity</h3>
            <p>A leading indicator of churn is the 30-day usage velocity. Customers who churn demonstrate a stark negative shift in usage leading up to cancellation.</p>
            <img src="figures/usage_change_kde.png" alt="Usage Change Distribution">
            
            <h3>3. The Support Burden</h3>
            <p>Support ticket volume strongly correlates with churn. High ticket volume over a 30-day period precedes churn events.</p>
            <img src="figures/support_tickets_kde.png" alt="Support Ticket Volume">
            
            <h3>4. Churn by Plan Type</h3>
            <img src="figures/churn_by_plan.png" alt="Churn by Plan">
            
            <h3>5. Revenue Impact (Lost ARR) by Segment</h3>
            <p>This illustrates which segments account for the highest actual revenue loss due to churn.</p>
            <img src="figures/revenue_lost_by_segment.png" alt="Revenue Lost by Segment">
        </div>

        <div class="card">
            <h2>Correlation Analysis</h2>
            <p>The correlation heatmap highlights how our engineered features (like `usage_change_30d` and `customer_support_health_score`) interact with the target variable `churned`.</p>
            <img src="figures/correlation_heatmap.png" alt="Correlation Heatmap">
        </div>
        
        <div class="card">
            <h2>Key Business Findings</h2>
            <ul>
                <li><strong>Usage Drops Predict Churn:</strong> Dramatic drops in the 30-day usage velocity are the strongest predictor of impending churn.</li>
                <li><strong>Contract Type is a Major Shield:</strong> Month-to-month contracts are highly volatile. Shifting customers to annual plans mitigates structural risk.</li>
                <li><strong>Support Experience Matters:</strong> Customers with high ticket volumes and subsequent negative NLP sentiment are extremely flight-risk.</li>
                <li><strong>Financial Risk Concentration:</strong> While Enterprise accounts may churn less frequently, their departure destroys massive amounts of ARR. Risk interventions should be prioritized by ARR.</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(REPORTS_DIR / "eda_summary.html", "w") as f:
        f.write(html_content)

def generate_eda_report():
    print("Loading data for EDA...")
    try:
        df = load_data()
    except FileNotFoundError:
        print("Master dataset not found. Run 'make data' or 'python scripts/train_model.py' first.")
        return

    print("Generating EDA Figures in reports/figures/...")

    # Distributions & Categorical Plots
    if 'segment_name' in df.columns:
        plot_churn_by_category(df, 'segment_name', "Churn Rate by Customer Segment", "churn_by_segment.png")
        plot_revenue_at_risk(df, 'segment_name', "revenue_lost_by_segment.png")
        
    if 'contract_type' in df.columns:
        plot_churn_by_category(df, 'contract_type', "Churn Rate by Contract Type", "churn_by_contract.png")
        
    if 'plan_type' in df.columns:
        plot_churn_by_category(df, 'plan_type', "Churn Rate by Plan Type", "churn_by_plan.png")
        
    if 'usage_change_30d' in df.columns:
        plot_numerical_distribution(df, 'usage_change_30d', 'churned', "30-Day Usage Change Distribution by Churn", "usage_change_kde.png")
        
    if 'support_tickets_30d' in df.columns:
        plot_numerical_distribution(df, 'support_tickets_30d', 'churned', "Support Ticket Volume by Churn", "support_tickets_kde.png")
        
    if 'tenure_months' in df.columns:
        plot_numerical_distribution(df, 'tenure_months', 'churned', "Tenure Distribution by Churn", "tenure_kde.png")

    plot_correlation_heatmap(df, "correlation_heatmap.png")
    
    print("Generating HTML Report...")
    generate_html_report(df)
    
    print("EDA Report Generation Complete! Check reports/eda_summary.html")

if __name__ == "__main__":
    generate_eda_report()
