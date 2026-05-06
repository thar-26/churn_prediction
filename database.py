import sqlite3
import pandas as pd
import os

DB_PATH = "churn.db"

def create_database():
    print("Creating SQLite database...")
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Clean TotalCharges — has spaces, convert to float
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
    df['totalcharges'].fillna(0, inplace=True)
    
    # Convert churn to binary
    df['churn_binary'] = (df['churn'] == 'Yes').astype(int)
    
    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    
    # Write main table
    df.to_sql('customers', conn, if_exists='replace', index=False)
    print(f"Loaded {len(df)} customers into database")
    
    # Create useful views using SQL
    conn.execute("""
    CREATE VIEW IF NOT EXISTS churn_by_contract AS
    SELECT 
        contract,
        COUNT(*) as total_customers,
        SUM(churn_binary) as churned,
        ROUND(AVG(churn_binary) * 100, 2) as churn_rate_pct,
        ROUND(AVG(monthlycharges), 2) as avg_monthly_charges,
        ROUND(SUM(monthlycharges * churn_binary), 2) as monthly_revenue_at_risk
    FROM customers
    GROUP BY contract
    ORDER BY churn_rate_pct DESC
    """)
    
    conn.execute("""
    CREATE VIEW IF NOT EXISTS churn_by_tenure AS
    SELECT 
        CASE 
            WHEN tenure <= 12 THEN '0-12 months'
            WHEN tenure <= 24 THEN '13-24 months'
            WHEN tenure <= 48 THEN '25-48 months'
            ELSE '49+ months'
        END as tenure_group,
        COUNT(*) as total_customers,
        SUM(churn_binary) as churned,
        ROUND(AVG(churn_binary) * 100, 2) as churn_rate_pct,
        ROUND(AVG(monthlycharges), 2) as avg_monthly_charges
    FROM customers
    GROUP BY tenure_group
    ORDER BY churn_rate_pct DESC
    """)

    conn.execute("""
    CREATE VIEW IF NOT EXISTS high_risk_customers AS
    SELECT 
        customerid,
        gender,
        seniorcitizen,
        tenure,
        contract,
        monthlycharges,
        totalcharges,
        churn
    FROM customers
    WHERE 
        contract = 'Month-to-month'
        AND tenure < 12
        AND monthlycharges > 60
    ORDER BY monthlycharges DESC
    """)

    conn.execute("""
    CREATE VIEW IF NOT EXISTS revenue_summary AS
    SELECT
        COUNT(*) as total_customers,
        SUM(churn_binary) as total_churned,
        ROUND(AVG(churn_binary) * 100, 2) as overall_churn_rate,
        ROUND(SUM(monthlycharges), 2) as total_monthly_revenue,
        ROUND(SUM(monthlycharges * churn_binary), 2) as monthly_revenue_at_risk,
        ROUND(SUM(monthlycharges * churn_binary) * 12, 2) as annual_revenue_at_risk
    FROM customers
    """)
    
    conn.commit()
    conn.close()
    print("Database and views created successfully!")

def run_query(query):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_summary():
    return run_query("SELECT * FROM revenue_summary")

def get_churn_by_contract():
    return run_query("SELECT * FROM churn_by_contract")

def get_churn_by_tenure():
    return run_query("SELECT * FROM churn_by_tenure")

def get_high_risk():
    return run_query("SELECT * FROM high_risk_customers LIMIT 100")

def get_all_customers():
    return run_query("SELECT * FROM customers")

if __name__ == "__main__":
    create_database()
    
    print("\n── Revenue Summary ──")
    print(get_summary().to_string(index=False))
    
    print("\n── Churn by Contract ──")
    print(get_churn_by_contract().to_string(index=False))
    
    print("\n── Churn by Tenure ──")
    print(get_churn_by_tenure().to_string(index=False))
    
    print(f"\n── High Risk Customers ──")
    print(f"{len(get_high_risk())} high risk customers identified")