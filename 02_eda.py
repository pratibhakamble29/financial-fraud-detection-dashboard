import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading cleaned dataset...")

# Load cleaned dataset
df = pd.read_csv('../outputs/fraud_clean.csv')

print("Dataset loaded!")
print(df.shape)

# -----------------------------
# Fraud Distribution
# -----------------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x='isFraud',
    data=df
)

plt.title('Fraud vs Non-Fraud Transactions')

plt.savefig(
    '../outputs/fraud_distribution.png'
)

plt.show()

# -----------------------------
# Transaction Amount Distribution
# -----------------------------

plt.figure(figsize=(10,5))

sns.histplot(
    df['TransactionAmt'],
    bins=50,
    kde=True
)

plt.title('Transaction Amount Distribution')

plt.savefig(
    '../outputs/transaction_amount_distribution.png'
)

plt.show()

# -----------------------------
# Fraud by Product Type
# -----------------------------

if 'ProductCD' in df.columns:

    plt.figure(figsize=(8,5))

    sns.countplot(
        x='ProductCD',
        hue='isFraud',
        data=df
    )

    plt.title('Fraud by Product Type')

    plt.savefig(
        '../outputs/fraud_by_product.png'
    )

    plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------

numeric_df = df.select_dtypes(
    include=['number']
)

top_corr = numeric_df.corr()['isFraud'].abs().sort_values(
    ascending=False
).head(15)

top_features = top_corr.index

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_df[top_features].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title('Top Correlated Features with Fraud')

plt.savefig(
    '../outputs/correlation_heatmap.png'
)

plt.show()

print("EDA completed successfully!")