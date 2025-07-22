Project Overview

This project aims to improve fraud detection capabilities for Adey Innovations Inc. by developing machine learning models that analyze e-commerce and bank credit transactions. The solution focuses on:

    Advanced pattern recognition
    Geolocation analysis
    Real-time monitoring capabilities
    Explainable AI techniques

This phase establishes the data foundation for fraud detection across:

    E-commerce transactions (user behavior patterns)
    Credit card transactions (anomaly detection)
    Geolocation verification (IP analysis)

Data Pipeline Architecture

graph LR A[Raw Data] --> B[Data Cleaning] B --> C[Feature Engineering] C --> D[Exploratory Analysis] D --> E[Processed Datasets]
Project Structure

fraud_detection_project/ ├── data/ ├── notebooks/ ├── src/ ├── models/ └── reports/
Task 1: Data Analysis and Preprocessing
Data Loading & Inspection
Load datasets with memory optimization

dtypes = { 'purchase_value': 'float32', 'age': 'int8', 'class': 'boolean' }

fraud_data = pd.read_csv('Fraud_Data.csv', parse_dates=['signup_time', 'purchase_time'], dtype=dtypes)
Initial data quality check

print(f"Missing Values:\n{fraud_data.isna().sum()}") print(f"\nData Types:\n{fraud_data.dtypes}")
Datasets Processed

    E-commerce Transactions (Fraud_Data.csv)
    IP to Country Mapping (IpAddress_to_Country.csv)
    Credit Card Transactions (creditcard.csv)

Data Cleaning Steps
For E-commerce Data:

    ✅ Removed duplicate transactions
    ✅ Converted signup_time and purchase_time to datetime objects
    ✅ Dropped records with missing values
    ✅ Standardized categorical variables (source, browser, sex)

For Credit Card Data:

    ✅ Scaled transaction amounts using StandardScaler
    ✅ Verified no missing values in the dataset
    ✅ Preserved anonymized features (V1-V28)

Feature Engineering
E-commerce Features Added:

    Geolocation Features:
        Mapped IP addresses to countries
        Converted IP addresses to integer format for range matching

    Temporal Features:
        hour_of_day: Hour when purchase was made
        day_of_week: Day of week (0-6)
        time_since_signup: Hours between signup and purchase

    Behavioral Features:
        user_transaction_count: Number of transactions per user

Credit Card Features:

    No additional features created (V1-V28 already PCA-transformed)
    Scaled Amount feature for model compatibility
