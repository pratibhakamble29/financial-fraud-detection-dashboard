import pandas as pd

# Load risky cards file
df = pd.read_csv('../outputs/sql_risky_cards.csv')

# Mask customer card
df['customer_card'] = '514967******' + \
    df['customer_card'].astype(str).str[-4:]

# Save updated file
df.to_csv('../outputs/sql_risky_cards.csv', index=False)

print("Done!")