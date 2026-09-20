import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path to import configs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import RAW_DATA_DIR

np.random.seed(42)

NUM_CUSTOMERS = 5000

def generate_customers():
    print("Generating customers...")
    customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, NUM_CUSTOMERS + 1)]
    
    countries = ["USA", "UK", "Canada", "Germany", "France", "Australia", "India"]
    country_probs = [0.4, 0.15, 0.1, 0.1, 0.1, 0.05, 0.1]
    
    industries = ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "Education"]
    company_sizes = ["1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"]
    channels = ["Organic Search", "Paid Ads", "Referral", "Direct", "Event"]
    
    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "country": np.random.choice(countries, NUM_CUSTOMERS, p=country_probs),
        "industry": np.random.choice(industries, NUM_CUSTOMERS),
        "company_size": np.random.choice(company_sizes, NUM_CUSTOMERS),
        "acquisition_channel": np.random.choice(channels, NUM_CUSTOMERS),
    })
    
    # Introduce some missing values realistically
    customers.loc[np.random.choice(NUM_CUSTOMERS, 200, replace=False), 'industry'] = np.nan
    
    return customers

def generate_subscriptions(customers):
    print("Generating subscriptions...")
    plan_types = ["Starter", "Professional", "Enterprise"]
    contract_types = ["Month-to-Month", "Annual", "2-Year"]
    payment_methods = ["Credit Card", "Invoice", "PayPal"]
    
    # Base ARR by plan
    base_arr = {"Starter": 1200, "Professional": 6000, "Enterprise": 24000}
    
    plans = np.random.choice(plan_types, NUM_CUSTOMERS, p=[0.5, 0.35, 0.15])
    
    subs = pd.DataFrame({
        "customer_id": customers["customer_id"],
        "plan_type": plans,
        "contract_type": np.random.choice(contract_types, NUM_CUSTOMERS, p=[0.4, 0.5, 0.1]),
        "payment_method": np.random.choice(payment_methods, NUM_CUSTOMERS),
        "tenure_months": np.random.randint(1, 60, NUM_CUSTOMERS),
        "auto_renew": np.random.choice([True, False], NUM_CUSTOMERS, p=[0.7, 0.3]),
        "discount_percent": np.random.choice([0, 10, 20], NUM_CUSTOMERS, p=[0.7, 0.2, 0.1]),
    })
    
    # Calculate ARR with discount and noise
    subs["base_arr"] = subs["plan_type"].map(base_arr)
    noise = np.random.normal(1, 0.1, NUM_CUSTOMERS)
    subs["annual_revenue"] = subs["base_arr"] * (1 - subs["discount_percent"] / 100) * noise
    subs["monthly_charges"] = subs["annual_revenue"] / 12
    
    subs.drop(columns=["base_arr"], inplace=True)
    return subs

def generate_product_usage(customers, subs):
    print("Generating product usage...")
    # Base usage logic: Enterprise > Pro > Starter, Higher tenure = slightly higher usage
    
    plan_multiplier = subs["plan_type"].map({"Starter": 1.0, "Professional": 2.0, "Enterprise": 5.0})
    
    login_freq = np.clip(np.random.normal(10, 5, NUM_CUSTOMERS) * plan_multiplier, 1, 30)
    weekly_active_days = np.clip(np.random.normal(3, 1.5, NUM_CUSTOMERS), 0, 7)
    
    # We will engineer a feature later: Usage drop indicates churn
    # We'll create latent "churn risk" here to drive the data generation
    latent_risk = np.random.uniform(0, 1, NUM_CUSTOMERS)
    
    # Usage drops significantly for high risk
    usage_change_30d = np.where(latent_risk > 0.7, 
                                np.random.normal(-30, 15, NUM_CUSTOMERS), # Drop
                                np.random.normal(5, 10, NUM_CUSTOMERS))   # Stable/Grow
    
    usage_change_90d = usage_change_30d + np.random.normal(0, 5, NUM_CUSTOMERS)
    
    usage = pd.DataFrame({
        "customer_id": customers["customer_id"],
        "login_frequency": login_freq,
        "weekly_active_days": weekly_active_days,
        "monthly_active_users": np.clip(np.random.normal(5, 2, NUM_CUSTOMERS) * plan_multiplier, 1, 100),
        "feature_adoption_rate": np.clip(np.random.normal(0.5, 0.2, NUM_CUSTOMERS) + (1 - latent_risk)*0.2, 0, 1),
        "usage_change_30d": usage_change_30d,
        "usage_change_90d": usage_change_90d,
        "latent_risk": latent_risk # Hidden column to help generate other tables consistently
    })
    
    return usage

def generate_support_tickets(usage):
    print("Generating support tickets...")
    latent_risk = usage["latent_risk"]
    
    # High risk -> more tickets, more unresolved
    num_tickets_30d = np.where(latent_risk > 0.8, 
                               np.random.poisson(3, NUM_CUSTOMERS), 
                               np.random.poisson(0.5, NUM_CUSTOMERS))
    
    open_tickets = np.where(latent_risk > 0.85, 
                            np.random.randint(1, 4, NUM_CUSTOMERS), 
                            0)
                            
    avg_resolution = np.where(latent_risk > 0.7, 
                              np.random.normal(48, 24, NUM_CUSTOMERS), 
                              np.random.normal(12, 6, NUM_CUSTOMERS))
    
    tickets = pd.DataFrame({
        "customer_id": usage["customer_id"],
        "support_tickets_30d": num_tickets_30d,
        "support_tickets_90d": num_tickets_30d + np.random.poisson(1, NUM_CUSTOMERS),
        "open_tickets": open_tickets,
        "avg_resolution_hours": np.clip(avg_resolution, 1, 168),
        "escalation_count": np.where(latent_risk > 0.9, np.random.poisson(1, NUM_CUSTOMERS), 0)
    })
    return tickets

def generate_customer_interactions(usage):
    print("Generating customer interactions...")
    latent_risk = usage["latent_risk"]
    
    csat = np.clip(np.where(latent_risk > 0.7, 
                            np.random.normal(3, 1, NUM_CUSTOMERS), 
                            np.random.normal(4.5, 0.5, NUM_CUSTOMERS)), 1, 5)
                            
    nps = np.clip(np.where(latent_risk > 0.7, 
                           np.random.normal(4, 2, NUM_CUSTOMERS), 
                           np.random.normal(8, 1.5, NUM_CUSTOMERS)), 0, 10)
    
    interactions = pd.DataFrame({
        "customer_id": usage["customer_id"],
        "emails_opened": np.random.poisson(5, NUM_CUSTOMERS),
        "emails_clicked": np.random.poisson(1, NUM_CUSTOMERS),
        "csat_score": csat,
        "nps_score": nps
    })
    
    # Missing CSAT/NPS is very common in reality
    interactions.loc[np.random.choice(NUM_CUSTOMERS, int(NUM_CUSTOMERS * 0.4), replace=False), 'csat_score'] = np.nan
    interactions.loc[np.random.choice(NUM_CUSTOMERS, int(NUM_CUSTOMERS * 0.5), replace=False), 'nps_score'] = np.nan
    
    return interactions

def generate_ticket_text_data(usage):
    print("Generating individual ticket text records...")
    # This generates a separate table of individual tickets for NLP
    tickets_list = []
    
    issue_categories = ["Billing", "Technical", "Performance", "Account", "Feature Request", "Integration", "Security"]
    
    for _, row in usage.iterrows():
        cid = row['customer_id']
        risk = row['latent_risk']
        
        # Decide how many tickets this customer had in total history
        num_hist_tickets = np.random.poisson(2) + (3 if risk > 0.8 else 0)
        
        for i in range(num_hist_tickets):
            cat = np.random.choice(issue_categories)
            
            # Generate dummy text/sentiment based on risk
            if risk > 0.8:
                sentiment = np.random.choice(["Negative", "Neutral"], p=[0.8, 0.2])
                if cat == "Technical":
                    desc = np.random.choice(["System is always down, very frustrating.", "Can't log in since yesterday.", "Your API is too slow and throwing 500 errors."])
                elif cat == "Billing":
                    desc = np.random.choice(["I was overcharged on my last invoice.", "Cancel my subscription, too expensive.", "Why did my price go up?"])
                else:
                    desc = "Terrible experience, not working as expected."
            else:
                sentiment = np.random.choice(["Positive", "Neutral", "Negative"], p=[0.4, 0.5, 0.1])
                if cat == "Technical":
                    desc = np.random.choice(["How do I reset my password?", "Need help setting up SSO.", "Works great but I have a question on API limits."])
                elif cat == "Billing":
                    desc = np.random.choice(["Can you send me last month's receipt?", "Update credit card info.", "Good service."])
                else:
                    desc = "Just a general question about features."
                    
            tickets_list.append({
                "ticket_id": f"TKT-{np.random.randint(100000, 999999)}",
                "customer_id": cid,
                "issue_category": cat,
                "description": desc,
                "sentiment": sentiment,
                "priority": np.random.choice(["Low", "Medium", "High"], p=[0.5, 0.3, 0.2])
            })
            
    return pd.DataFrame(tickets_list)

def generate_churn_labels(usage, subs):
    print("Generating churn labels...")
    # Base probability on latent risk + some logical business rules
    risk = usage["latent_risk"].copy()
    
    # Modifiers
    # Month-to-month are more likely to churn
    risk += np.where(subs["contract_type"] == "Month-to-Month", 0.15, 0)
    # 2-Year are less likely
    risk -= np.where(subs["contract_type"] == "2-Year", 0.2, 0)
    # Auto-renew lowers risk
    risk -= np.where(subs["auto_renew"] == True, 0.1, 0)
    
    # Normalize prob to 0-1
    prob = np.clip(risk, 0.01, 0.99)
    
    # Actual churn based on probability
    churned = np.random.binomial(1, prob)
    
    labels = pd.DataFrame({
        "customer_id": usage["customer_id"],
        "churn_probability_ground_truth": prob,
        "churned": churned
    })
    
    # Assign churn date if churned
    base_date = datetime.now()
    labels["churn_date"] = [
        (base_date + timedelta(days=np.random.randint(1, 30))).strftime('%Y-%m-%d') if c == 1 else None 
        for c in churned
    ]
    
    return labels

def main():
    customers = generate_customers()
    subs = generate_subscriptions(customers)
    usage = generate_product_usage(customers, subs)
    tickets_agg = generate_support_tickets(usage)
    interactions = generate_customer_interactions(usage)
    nlp_tickets = generate_ticket_text_data(usage)
    labels = generate_churn_labels(usage, subs)
    
    # Drop latent_risk
    usage.drop(columns=["latent_risk"], inplace=True)
    
    # Save to raw
    customers.to_csv(RAW_DATA_DIR / "customers.csv", index=False)
    subs.to_csv(RAW_DATA_DIR / "subscriptions.csv", index=False)
    usage.to_csv(RAW_DATA_DIR / "product_usage.csv", index=False)
    tickets_agg.to_csv(RAW_DATA_DIR / "support_tickets.csv", index=False)
    interactions.to_csv(RAW_DATA_DIR / "customer_interactions.csv", index=False)
    nlp_tickets.to_csv(RAW_DATA_DIR / "nlp_tickets.csv", index=False)
    labels.to_csv(RAW_DATA_DIR / "churn_labels.csv", index=False)
    
    print("Data generation complete! Files saved to data/raw/")
    
    # Overall Churn Rate
    print(f"Overall Churn Rate: {labels['churned'].mean():.2%}")

if __name__ == "__main__":
    main()
