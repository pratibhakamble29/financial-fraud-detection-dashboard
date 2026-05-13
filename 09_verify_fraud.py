import pandas as pd

df = pd.read_csv('../outputs/fraud_engineered.csv')

# Check values
print("isFraud unique values:", df['isFraud'].unique())
print("isFraud dtype:", df['isFraud'].dtype)

print("\nFraud rows:", len(df[df['isFraud']==1]))
print("Legit rows:", len(df[df['isFraud']==0]))

# Fraud amount
fraud_amt = df[df['isFraud']==1]['TransactionAmt'].sum()
total_amt = df['TransactionAmt'].sum()

print(f"\nActual Fraud Amount: ${fraud_amt:,.2f}")
print(f"Total Amount: ${total_amt:,.2f}")