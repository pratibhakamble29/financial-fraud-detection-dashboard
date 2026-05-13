import pandas as pd
import numpy as np

print("Loading datasets...")

# Load sample data
sample_size = 100000

df_trans = pd.read_csv(
    '../data/train_transaction.csv',
    nrows=sample_size
)

df_id = pd.read_csv(
    '../data/train_identity.csv',
    nrows=sample_size
)


print("Datasets loaded successfully!")

# Merge datasets
df = df_trans.merge(
    df_id,
    on='TransactionID',
    how='left'
)

print("Datasets merged!")
print(df.shape)

# Missing value percentage
missing_percent = (
    df.isnull().mean() * 100
)

# Drop columns with >90% missing
high_missing_cols = missing_percent[
    missing_percent > 90
].index

df.drop(
    columns=high_missing_cols,
    inplace=True
)

print(f"Dropped {len(high_missing_cols)} high-missing columns")

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print("Duplicates removed!")

# Fill numeric missing values
numeric_cols = df.select_dtypes(
    include=['number']
).columns

for col in numeric_cols:
    df[col] = df[col].fillna(
        df[col].median()
    )

# Fill categorical missing values
cat_cols = df.select_dtypes(
    include=['object']
).columns

for col in cat_cols:
    df[col] = df[col].fillna(
        'Unknown'
    )

print("Missing values handled!")

# Save cleaned dataset
df.to_csv(
    '../outputs/fraud_clean.csv',
    index=False
)

print("Cleaned dataset saved successfully!")
print(df.head())