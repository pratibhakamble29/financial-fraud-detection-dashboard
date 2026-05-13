import pandas as pd
import sqlite3

print("Loading engineered dataset...")

# Load dataset
df = pd.read_csv('../outputs/fraud_engineered.csv')

print("Dataset loaded!")
print(df.shape)

# -----------------------------------
# Create SQLite Database
# -----------------------------------

conn = sqlite3.connect(':memory:')

df.to_sql(
    'transactions',
    conn,
    index=False,
    if_exists='replace'
)

print("Database created successfully!")

# -----------------------------------
# Query 1 - Top Risky Customers
# -----------------------------------

query1 = """
SELECT
    card1,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS fraud_transactions,
    ROUND(
        SUM(isFraud) * 100.0 / COUNT(*),
        2
    ) AS fraud_percentage
FROM transactions
GROUP BY card1
HAVING COUNT(*) >= 5
ORDER BY fraud_percentage DESC
LIMIT 10;
"""

top_risky = pd.read_sql_query(query1, conn)

print("\n===== Top Risky Customers =====")
print(top_risky)

# -----------------------------------
# Query 2 - Fraud by Product Type
# -----------------------------------

query2 = """
SELECT
    ProductCD,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS fraud_transactions,
    ROUND(
        SUM(isFraud) * 100.0 / COUNT(*),
        2
    ) AS fraud_percentage
FROM transactions
GROUP BY ProductCD
ORDER BY fraud_percentage DESC;
"""

fraud_product = pd.read_sql_query(query2, conn)

print("\n===== Fraud by Product Type =====")
print(fraud_product)

# -----------------------------------
# Query 3 - Daily Fraud Trend
# -----------------------------------

query3 = """
SELECT
    CAST(TransactionDT / 86400 AS INT) AS transaction_day,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS fraud_transactions
FROM transactions
GROUP BY transaction_day
ORDER BY transaction_day;
"""

daily_trend = pd.read_sql_query(query3, conn)

print("\n===== Daily Fraud Trend =====")
print(daily_trend.head())

# -----------------------------------
# Query 4 - High Risk Transactions
# -----------------------------------

query4 = """
SELECT
    TransactionID,
    TransactionAmt,
    RiskScore,
    isFraud
FROM transactions
WHERE RiskScore >= 50
ORDER BY TransactionAmt DESC
LIMIT 10;
"""

high_risk = pd.read_sql_query(query4, conn)

print("\n===== High Risk Transactions =====")
print(high_risk)

print("\nSQL Analysis completed successfully!")