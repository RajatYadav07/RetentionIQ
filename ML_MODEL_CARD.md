# Model Card: RetentionIQ Churn Predictor

## Model Details
- **Model Type**: XGBoost Classifier
- **Objective**: Predict customer churn within the next 30 days.
- **Handling Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique) applied to training data.

## Features
- Demographic (Country, Industry, Size)
- Product Usage (Login frequency, feature adoption, 30d/90d change)
- Support (Tickets in last 30d, open tickets, resolution time, sentiment score)
- Financial (Tenure, ARR, Plan type)

## Evaluation Metrics (Target)
Because churn is imbalanced, accuracy is misleading. The model is optimized and evaluated on:
- **PR-AUC** (Precision-Recall Area Under Curve)
- **Recall** (Identifying as many at-risk customers as possible)
- **F1-Score**

## Ethical Considerations & Limitations
- The model relies heavily on usage changes. It may not predict "sudden" churn caused by external market factors or competitor pricing if usage hasn't dropped yet.
- Sentiment analysis uses VADER, which might misinterpret sarcasm or highly technical support complaints.
