# Model Card: RetentionIQ Risk Predictor (v1.1.0)

## Model Details
- **Architecture**: Unified Scikit-Learn Pipeline
  - Preprocessing: Standardization & One-Hot Encoding
  - Class Balancing: SMOTE (applied strictly on training set during fit)
  - Base Estimator: XGBoost Classifier
  - Calibration: `CalibratedClassifierCV` (Sigmoid method, 3-fold CV)
- **Objective**: Predict the true probability of customer churn within the next 30 days.

## Intended Use
- **Primary Use Case**: Ranking and prioritizing customer success outreach based on churn probability and revenue at risk.
- **Secondary Use Case**: Understanding global and local risk drivers (e.g. usage drops, high support ticket volumes) via SHAP values to inform product and support strategy.

## Limitations & Assumptions
- **Synthetic Correlations**: This model was trained on a generated dataset. While it attempts to mimic realistic enterprise SaaS dynamics (e.g. `usage_change_ratio` inversely correlated with churn), it may encode synthetic biases or artifacts not perfectly representative of a specific real-world company.
- **Stationarity**: The model assumes customer behavior remains relatively static. Sudden external market shocks (e.g., a competitor launching a cheaper product) will not be captured unless it immediately manifests in usage metrics.

## Possible Biases
- **Support-Heavy Segments**: The model heavily weights open support tickets and escalation counts. Customers in complex deployments naturally have higher ticket volumes but may not necessarily be a churn risk. The model might over-flag enterprise customers who are heavily engaged with support for feature requests rather than bugs.

## Error Analysis
- **False Positives (Precision drops)**: Flagging a healthy customer as at-risk wastes Customer Success Manager (CSM) time. We mitigate this by optimizing the decision threshold for F1-score rather than pure recall, and calculating the exact ARR at risk to prioritize effectively.
- **False Negatives (Recall drops)**: Failing to flag a churner results in lost revenue. If a customer churns purely due to budget cuts without altering their login patterns beforehand, the model will likely yield a false negative.

## Feature List
Engineered Features include:
- `usage_change_ratio` (30d vs 90d velocity)
- `support_ticket_density` (tickets normalized by tenure)
- `is_contract_ending_soon`
- Standard Firmographics (Country, Industry, Size)

## Threshold Selection Strategy
Because predicting churn suffers from severe class imbalance and asymmetric business costs, the model does not use the default `0.5` probability threshold. 
During training, the pipeline iterates through the Precision-Recall curve to find the optimal threshold that maximizes the **F1-Score**. This threshold is saved in the model metadata and applied during inference.
