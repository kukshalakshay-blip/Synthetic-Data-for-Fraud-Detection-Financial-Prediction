import os
import pandas as pd
from synthetic_gen import generate_synthetic_data

def main():
    os.makedirs('data/synthetic', exist_ok=True)

    # 1. Synthesize Fraud Data
    print("Synthesizing Fraud Data...")
    try:
        fraud_train = pd.read_csv('data/processed/fraud_train.csv')
        # Generate equivalent number of rows to training set
        fraud_synth = generate_synthetic_data(fraud_train, num_rows=len(fraud_train), epochs=300) # Increased epochs for quality
        fraud_synth.to_csv('data/synthetic/fraud_synth.csv', index=False)
        print("Successfully generated and saved data/synthetic/fraud_synth.csv")
    except Exception as e:
        print(f"Error synthesizing Fraud Data: {e}")

    # 2. Synthesize SPY Data
    print("\nSynthesizing Financial Data (SPY)...")
    try:
        spy_train = pd.read_csv('data/processed/spy_train.csv')
        spy_synth = generate_synthetic_data(spy_train, num_rows=len(spy_train), epochs=300) # Increased epochs for quality
        spy_synth.to_csv('data/synthetic/spy_synth.csv', index=False)
        print("Successfully generated and saved data/synthetic/spy_synth.csv")
    except Exception as e:
        print(f"Error synthesizing Financial Data: {e}")

if __name__ == '__main__':
    main()
