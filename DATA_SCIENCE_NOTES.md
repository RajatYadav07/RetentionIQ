# Data Science Analytical Notes & Assumptions

This document summarizes the findings from our Exploratory Data Analysis (`src/data/eda.py`) and outlines the logic behind the advanced feature engineering in the ML pipeline.

## 1. Key Exploratory Findings

### A. The Impact of Declining Usage
- **Finding**: Customers exhibiting a decline in product usage over the last 30 days have a **61.5% churn rate**, compared to just **32.2%** for customers with stable or growing usage.
- **Action**: We engineered velocity features (`usage_change_30d`, `usage_change_90d`) and a `days_since_last_activity` proxy to heavily capture this signal.

### B. The Support Burden Threshold
- **Finding**: The relationship between support tickets and churn is non-linear. 
  - 0-2 tickets: ~40-47% churn.
  - 3-5 tickets: **79.4% churn**.
  - 5+ tickets: **82.1% churn**.
- **Action**: We introduced `complaint_ratio` (open vs total) and `support_severity_score` (which multiplies resolution hours by escalations) to help the model distinguish between a "healthy" support interaction (feature request) and a "critical" one (system outage).

### C. NLP Sentiment as a Leading Indicator
- **Finding**: We aggregated individual support ticket text sentiment. Customers averaging a **Negative** sentiment score have a **72.9% churn rate**, whereas **Positive** and **Neutral** hover around ~37-38%.
- **Action**: The pipeline now automatically maps unstructured ticket sentiment into a structured `sentiment_trend` feature.

### D. Contract Structural Risks
- **Finding**: **Month-to-Month** contracts suffer from a **58.2% churn rate**, nearly triple the **23.8%** rate seen in 2-Year commitments.
- **Action**: We engineered `contract_days_remaining` to dynamically calculate the time pressure on a renewal decision.

## 2. Advanced Feature Definitions & Assumptions

We implemented 10 advanced business features directly into the `scikit-learn` unified pipeline using a stateless `FunctionTransformer`. This ensures that **target leakage is physically impossible** since transformations happen strictly within the cross-validation/inference step on a row-by-row basis.

| Feature | Logic / Formula | Business Rationale |
|---------|-----------------|--------------------|
| `usage_change_30d` | Native / Standardized | Direct proxy for recent health. |
| `usage_change_90d` | Native / Standardized | Captures medium-term historical baseline. |
| `support_ticket_velocity` | `30d_tickets / (90d_tickets / 3)` | Identifies sudden spikes in support requests. |
| `complaint_ratio` | `open_tickets / 30d_tickets` | Measures how much of the recent support burden is unresolved. |
| `engagement_score` | Weighted sum of logins (40%), active days (30%), feature adoption (30%) | Creates a unified metric for overall product stickiness. |
| `customer_value_score` | `ARR * Tenure * (1 - Discount)` | A proxy for Customer Lifetime Value (CLV) to prioritize high-worth accounts. |
| `days_since_last_activity`| `30 / login_frequency` (Capped at 60) | Inferred recency metric. Low logins = high days since last seen. |
| `contract_days_remaining` | Modulo arithmetic on `tenure` based on contract length | Identifies customers entering the "renewal risk window". |
| `support_severity_score` | `avg_resolution_hours * (escalations + 1)` | Penalizes accounts that experience severe, slow-to-resolve outages. |
| `sentiment_trend` | Mean sentiment score (-1 to 1) | Translates raw NLP classification into a continuous numerical risk feature. |

## 3. Assumptions & Limitations
- **Data Completeness**: We assume that `login_frequency` is an accurate reflection of daily active usage. In environments where users stay logged in for weeks, this feature might under-represent engagement. We fall back on `feature_adoption_rate` to balance the `engagement_score`.
- **Value Scoring**: The `customer_value_score` does not account for variable acquisition costs across channels (e.g., Paid Ads vs Organic), so it is a gross revenue metric, not a net margin metric.
