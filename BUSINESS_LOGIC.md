# Business Logic

## Risk Scoring
- **HIGH RISK**: Churn Probability >= 0.70
- **MEDIUM RISK**: Churn Probability >= 0.40
- **LOW RISK**: Churn Probability < 0.40

## Recommendation Engine Rules
The recommendation engine layers business context on top of ML probability:

1. **High Value + Negative Sentiment + High Risk**
   - Action: Priority Customer Success intervention
   - Urgency: Critical
2. **High Support Volume + High Risk**
   - Action: Technical escalation & Support review
3. **Significant Usage Drop (< -20%) + High Risk**
   - Action: Product adoption campaign
4. **No Discount + High Risk**
   - Action: Retention discount / plan review
5. **Month-to-Month Contract + High Risk**
   - Action: Renewal outreach / Incentivize annual plan

## Revenue at Risk
Calculated as: `Annual Revenue * Churn Probability` for customers above the 0.5 decision threshold.
