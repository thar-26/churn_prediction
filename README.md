\# Customer Churn Prediction \& Business Intelligence Dashboard



End-to-end churn analysis system identifying $1.67M annual revenue at risk 

from a telecom customer base of 7,043. Built with SQL, Python, XGBoost, 

SHAP explainability, and a live Streamlit dashboard.



\---



\## What it does



\*\*SQL Analytics Layer\*\*

SQLite database with analytical views covering churn by contract type, 

tenure cohorts, high-risk customer segments, and revenue at risk — 

all queryable directly from Python or any BI tool.



\*\*Machine Learning Model\*\*

XGBoost classifier achieving AUC-ROC of 0.838 on churn prediction.

SHAP explainability reveals contract type and charge-per-tenure ratio 

as the top two drivers — actionable insights, not just a black box.



\*\*Business Intelligence Dashboard\*\*

Live Streamlit dashboard showing KPI cards, churn by segment, 

risk distribution, top high-risk customers, and a real-time 

churn probability checker for any customer profile.



\---



\## Key findings



\- $1.67M annual revenue at risk from churning customers

\- Month-to-month customers churn at 42.71% vs 2.83% on two-year contracts

\- 47.44% of new customers (0-12 months) churn — highest risk window

\- Contract type is the single biggest churn driver (SHAP analysis)

\- 975 high-risk customers identified for immediate retention outreach



\---



\## How to run



```bash

git clone https://github.com/thar-26/churn\_prediction

cd churn\_prediction

pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn plotly streamlit sqlalchemy

```



Download the dataset from Kaggle:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn



Place `WA\_Fn-UseC\_-Telco-Customer-Churn.csv` in the folder, then:



```bash

python database.py

python model.py

streamlit run app.py

```



\---



\## Stack

Python · SQL (SQLite) · XGBoost · SHAP · Pandas · Plotly · Streamlit

