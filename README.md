# Customer Retention Risk Modeling — Lloyds Banking Group

This project aims to identify customers at risk of churning from Lloyds Banking Group by engineering a unified RFM behavioral dataset and developing a machine learning classification pipeline capable of flagging at-risk customers before attrition occurs.

![Logo] (https://www.reinsurancene.ws/wp-content/uploads/2025/06/lloyds-bank-768x480.jpg)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)

---

## 📊 **Project Overview**

This project covers an end-to-end data science workflow including:

- Integrating 5 disparate customer data sources into a unified RFM Master Table.
- Conducting Exploratory Data Analysis (EDA) and statistical significance testing.
- Engineering behavioral features such as Login Consistency and Spend Efficiency.
- Handling class imbalance using `class_weight='balanced'` in the classifier.
- Training and evaluating a Random Forest classification pipeline with RandomizedSearchCV and GridSearchCV tuning.
- Delivering threshold-based retention intervention strategies for business stakeholders.

---

## 📂 **Dataset**

The dataset consists of **1,000 synthetic customer records** across 5 integrated tables prepared for Lloyds Banking Group's Data Science Division:

| Dataset | Key Features |
|---|---|
| Demographics | Age, Gender, IncomeLevel, MaritalStatus |
| Transaction History | Total_Spend, Transaction_Count |
| Online Activity | LastLoginDate, Total_Logins, ServiceUsage |
| Customer Service | Interaction_Count, Unresolved_Issues |
| Churn Status | ChurnStatus (1 = Churned, 0 = Retained) |

The dataset exhibits a **20.4% churn rate**, providing a sufficient signal-to-noise ratio for supervised learning. No missing values or extreme outliers were identified during preprocessing.

---

## 🛠️ **Features**

- End-to-end data pipeline from raw ingestion to model-ready RFM master table.
- Statistical significance testing (Chi-Square tests, T-tests, Pearson correlation heatmaps).
- Engineered features: `Recency_Login`, `Login_Consistency`, `Spend_Efficiency`, `Most_Frequent_Product_Category`.
- One-Hot Encoding for categorical variables and StandardScaler for numerical standardization.
- RandomizedSearchCV and GridSearchCV hyperparameter tuning with 5-fold cross-validation.
- Risk-score segmentation for business retention strategy deployment.

---

## 🚀 **Key Insights**

- **Digital Engagement** is the primary churn driver — retained customers averaged **26.49 logins** vs. **23.65** for churners (p = 0.0129).
- **Age Risk Zones** — customers aged **18–35** and **50–58** show the highest churn propensity.
- **Spend Loyalty Band** — customers spending between **$600–$1,300** exhibit the strongest retention rates.
- **Seasonal Patterns** — churn spikes identified in **October and August**, suggesting external market pressures or contract cycle sensitivity.
- **Complaint Friction** — while most service interactions are resolved, complaints remain disproportionately unresolved, representing a key attrition risk.
- **Traditional demographics (Age, Gender, Income) showed statistical neutrality** — behavioral markers outperform demographic profiling for churn prediction.

---

## 📌 **Recommendations**

- Implement **"Nudge" push notifications** for users inactive on the mobile app for more than **15 days**.
- Automatically funnel customers with a risk score **> 0.35** into personalized email campaigns or targeted interest rate offers.
- Assign relationship managers for direct outreach to high-value "Whale" customers with a risk score **> 0.40**.
- Prioritize complaint resolution workflows to reduce unresolved interaction rates.
- Investigate seasonal retention incentives ahead of **October and August** churn spikes.

---

## 🎯 **Results**

| Metric | Score | Interpretation |
|---|---|---|
| Accuracy | 0.63 | Overall correctness of the model |
| Recall (Churn) | 0.37 | 37% of actual churners successfully identified |
| Precision | 0.24 | 23% of high-risk flags confirmed as churners |
| ROC-AUC | 0.51 | Baseline classifier — improvement targeted via external data integration |

**Confusion Matrix:**
- True Positives: **15** — customers correctly flagged for intervention
- False Positives: **48** — loyal customers receiving harmless retention offers
- False Negatives: **26** — primary area for future improvement

---

## 🔮 **Future Improvements**

- **External Data Integration** — incorporate Bank of England base rate changes and competitor proximity signals to improve ROC-AUC beyond the current baseline.
- **Deep Learning** — explore LSTM or temporal Neural Networks to capture sequential transaction patterns that ensemble trees overlook.
- **Real-Time Feature Stores** — deploy a real-time feature engineering layer to react to behavioral triggers (e.g., sudden large withdrawals) within seconds.

