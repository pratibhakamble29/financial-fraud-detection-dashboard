import pandas as pd
import sqlite3

print("Loading engineered dataset...")

# Load dataset
df = pd.read_csv('../outputs/fraud_engineered.csv')

# Create database
conn = sqlite3.connect(':memory:')

df.to_sql(
    'transactions',
    conn,
    index=False,
    if_exists='replace'
)

print("Database ready!")

# -----------------------------------
# Top Risky Customers
# -----------------------------------

query1 = """
SELECT
    card1 AS customer_card,
    COUNT(*) AS total_txns,
    SUM(isFraud) AS fraud_txns,
    ROUND(
        SUM(isFraud) * 100.0 / COUNT(*),
        2
    ) AS fraud_rate_pct,
    ROUND(
        AVG(TransactionAmt),
        2
    ) AS avg_txn_amt
FROM transactions
GROUP BY card1
HAVING COUNT(*) >= 5
ORDER BY fraud_rate_pct DESC
LIMIT 10;
"""

risky_cards = pd.read_sql_query(
    query1,
    conn
)

risky_cards.to_csv(
    '../outputs/sql_risky_cards.csv',
    index=False
)

print("sql_risky_cards.csv exported!")

# -----------------------------------
# Email Domain Fraud Analysis
# -----------------------------------

if 'P_emaildomain' in df.columns:

    query2 = """
    SELECT
        P_emaildomain AS email_domain,
        ROUND(
            SUM(TransactionAmt),
            2
        ) AS total_fraud_value
    FROM transactions
    WHERE isFraud = 1
    GROUP BY P_emaildomain
    ORDER BY total_fraud_value DESC
    LIMIT 10;
    """

    email_fraud = pd.read_sql_query(
        query2,
        conn
    )

    email_fraud.to_csv(
        '../outputs/sql_email_fraud.csv',
        index=False
    )

    print("sql_email_fraud.csv exported!")

print("Export completed successfully!")