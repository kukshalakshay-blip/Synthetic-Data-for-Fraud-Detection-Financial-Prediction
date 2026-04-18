import os
import pandas as pd
from data_processing import load_and_preprocess_fraud_data, load_and_preprocess_financial_data

def main():
    os.makedirs('data/processed', exist_ok=True)
    
    print("Processing Fraud Data...")
    try:
        X_train_f, X_test_f, y_train_f, y_test_f, raw_df_f = load_and_preprocess_fraud_data(
            filepath='data/raw/creditcard.csv',
            sample_frac=1.0 # Using 1.0 for full pipeline
        )
        # Combine into cohesive training and testing sets
        fraud_train = pd.concat([X_train_f, y_train_f], axis=1)
        fraud_test = pd.concat([X_test_f, y_test_f], axis=1)
        
        fraud_train.to_csv('data/processed/fraud_train.csv', index=False)
        fraud_test.to_csv('data/processed/fraud_test.csv', index=False)
        print("Fraud data successfully processed and saved to data/processed/")
    except Exception as e:
        print(f"Error processing Fraud Data: {e}")

    print("\nProcessing Financial Data (SPY)...")
    try:
        X_train_ts, X_test_ts, y_train_ts, y_test_ts, raw_df_ts = load_and_preprocess_financial_data(
            filepath='data/raw/SPY_daily.csv'
        )
        spy_train = pd.concat([X_train_ts, y_train_ts], axis=1)
        spy_test = pd.concat([X_test_ts, y_test_ts], axis=1)
        
        spy_train.to_csv('data/processed/spy_train.csv', index=False)
        spy_test.to_csv('data/processed/spy_test.csv', index=False)
        print("Financial data successfully processed and saved to data/processed/")
    except Exception as e:
        print(f"Error processing Financial Data: {e}")

if __name__ == '__main__':
    # Script should be run from src/
    main()
