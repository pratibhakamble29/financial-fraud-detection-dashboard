import pandas as pd
import numpy as np

print("Loading cleaned dataset...")

# Load cleaned dataset
df = pd.read_csv('../outputs/fraud_clean.csv')

print("Dataset loaded!")
print(df.shape)

# -----------------------------------
# Transaction Hour Feature
# -----------------------------------

df['TransactionHour'] = (
    df['TransactionDT'] // 3600
) % 24

print("TransactionHour feature created!")

# -----------------------------------
# Night Transaction Flag
# -----------------------------------

df['IsNightTransaction'] = df[
    'TransactionHour'
].apply(
    lambda x: 1 if x >= 22 or x <= 5 else 0
)

print("Night transaction feature created!")

# -----------------------------------
# User Transaction Count
# -----------------------------------

df['UserTransactionCount'] = df.groupby(
    'card1'
)['TransactionID'].transform('count')

print("User transaction count feature created!")

# -----------------------------------
# Average Transaction Amount Per User
# -----------------------------------

df['AvgTransactionAmountPerUser'] = df.groupby(
    'card1'
)['TransactionAmt'].transform('mean')

print("Average transaction feature created!")

# -----------------------------------
# Amount Difference From User Average
# -----------------------------------

df['AmountDeviation'] = (
    df['TransactionAmt']
    - df['AvgTransactionAmountPerUser']
)

print("Amount deviation feature created!")

# -----------------------------------
# High Amount Transaction Flag
# -----------------------------------

high_amount_threshold = df[
    'TransactionAmt'
].quantile(0.95)

df['HighAmountFlag'] = df[
    'TransactionAmt'
].apply(
    lambda x: 1 if x > high_amount_threshold else 0
)

print("High amount flag created!")

# -----------------------------------
# Risk Score
# -----------------------------------

df['RiskScore'] = (
    df['HighAmountFlag'] * 30
    + df['IsNightTransaction'] * 20
)

print("Risk score feature created!")

# -----------------------------------
# Save Engineered Dataset
# -----------------------------------

df.to_csv(
    '../outputs/fraud_engineered.csv',
    index=False
)

print("Feature engineering completed!")
print("Engineered dataset saved!")
print(df.head())