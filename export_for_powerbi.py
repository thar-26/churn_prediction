import sqlite3
import pandas as pd

conn = sqlite3.connect("churn.db")

# Export all tables Power BI needs
tables = {
    "customers_scored": "customers_scored.csv",
    "churn_by_contract": "churn_by_contract.csv",
    "churn_by_tenure": "churn_by_tenure.csv",
    "revenue_summary": "revenue_summary.csv",
    "high_risk_customers": "high_risk_customers.csv"
}

for table, filename in tables.items():
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    df.to_csv(filename, index=False)
    print(f"Exported {filename} — {len(df)} rows")

conn.close()
print("All files exported for Power BI!")