import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('transaction_data.csv')

# Ensure date columns are in datetime format
data['post_date'] = pd.to_datetime(data['post_date'], errors='coerce')
data['effective_date'] = pd.to_datetime(data['effective_date'], errors='coerce')

# Derive features
# 1. Transaction Amount (assuming 'total' column represents this)
data['transaction_amount'] = data['total']

# 2. Transaction Date and Time
data['transaction_date'] = data['post_date']

# 3. Transaction Frequency (number of transactions in the past 30 days)
data['transaction_frequency'] = data.groupby('account number')['post_date'].transform(
    lambda x: x.rolling('30D', on='post_date').count())

# 4. Account Age (days since the first transaction)
data['account_age'] = data.groupby('account number')['post_date'].transform(
    lambda x: (x - x.min()).dt.days)

# 5. Historical Transaction Data (average transaction amount)
data['avg_transaction_amount'] = data.groupby('account number')['transaction_amount'].transform('mean')

# 6. Transaction Velocity (number of transactions in the past day)
data['transaction_velocity'] = data.groupby('account number')['post_date'].transform(
    lambda x: x.rolling('1D', on='post_date').count())

# 7. Behavioral Biometrics (not directly derivable from given data)

# 8. Account Balance (assuming 'principle' column represents the balance)
data['account_balance'] = data['principle']

# 9. Geo-Velocity (not derivable without location data)

# Select relevant columns
features = [
    'transaction_amount', 'transaction_date', 'transaction_frequency',
    'account_age', 'avg_transaction_amount', 'transaction_velocity',
    'account_balance'
]

# Drop rows with missing values in selected features
feature_data = data[features].dropna()

# Save to CSV
feature_data.to_csv('derived_features.csv', index=False)
#Explanation
#Loading Data: The data is loaded from a CSV file.
#Date Conversion: The post_date and effective_date columns are converted to datetime format.
#Feature Derivation:
#Transaction Amount: Assumed to be the 'total' column.
#Transaction Date: Directly from the 'post_date' column.
#Transaction Frequency: Calculated as the number of transactions in the past 30 days for each account.
#Account Age: Calculated as the number of days since the first transaction for each account.
#Historical Transaction Data: Calculated as the average transaction amount for each account.
#Transaction Velocity: Calculated as the number of transactions in the past day for each account.
#Account Balance: Assumed to be the 'principle' column.
#Behavioral Biometrics and Geo-Velocity: Not derivable from the provided data.










