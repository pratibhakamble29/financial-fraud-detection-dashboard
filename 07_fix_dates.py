import pandas as pd

df = pd.read_csv('../outputs/fraud_engineered.csv')

# Convert seconds into real datetime
df['TransactionDate'] = pd.to_datetime('2017-11-30') + \
    pd.to_timedelta(df['TransactionDT'], unit='s')

# Keep only date
df['TransactionDate'] = pd.to_datetime(
    df['TransactionDate'].dt.strftime('%Y-%m-%d')
)

# Check
print("Min date:", df['TransactionDate'].min())
print("Max date:", df['TransactionDate'].max())
print("Sample:\n", df['TransactionDate'].head())

# Save updated dataset
df.to_csv('../outputs/fraud_engineered.csv', index=False)

print("✅ Saved!")