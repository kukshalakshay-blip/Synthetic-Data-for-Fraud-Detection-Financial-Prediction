import os
import pandas as pd
import joblib
from models.baseline import train_evaluate_xgb, train_evaluate_rf
from models.baseline_ts import train_evaluate_xgb_ts, train_evaluate_rf_ts

def train_and_save(domain_name, train_df, synth_df, test_df, target_col, train_func):
    """Utility to train on REAL and SYNTHETIC data and save the Joblib outputs."""
    X_test, y_test = test_df.drop(target_col, axis=1), test_df[target_col]
    
    # 1. Train on real data
    X_train_r, y_train_r = train_df.drop(target_col, axis=1), train_df[target_col]
    clf_real, metrics_real = train_func(X_train_r, y_train_r, X_test, y_test)
    joblib.dump(clf_real, f'models/{domain_name}_xgb_real.joblib')
    print(f"[{domain_name} Real] Accuracy: {metrics_real['accuracy']:.4f}")
    
    # 2. Train on synthetic data
    X_train_s, y_train_s = synth_df.drop(target_col, axis=1), synth_df[target_col]
    clf_synth, metrics_synth = train_func(X_train_s, y_train_s, X_test, y_test)
    joblib.dump(clf_synth, f'models/{domain_name}_xgb_synth.joblib')
    print(f"[{domain_name} Synthetic] Accuracy: {metrics_synth['accuracy']:.4f}")

def main():
    os.makedirs('models', exist_ok=True)
    
    # Train Fraud Models
    print("Training Fraud Models (XGBoost)...")
    try:
        f_train = pd.read_csv('data/processed/fraud_train.csv')
        f_test = pd.read_csv('data/processed/fraud_test.csv')
        f_synth = pd.read_csv('data/synthetic/fraud_synth.csv')
        
        train_and_save("fraud", f_train, f_synth, f_test, "Class", train_evaluate_xgb)
    except Exception as e:
        print(f"Error training Fraud models: {e}")

    # Train Finance Models
    print("\nTraining Finance Models (XGBoost)...")
    try:
        s_train = pd.read_csv('data/processed/spy_train.csv')
        s_test = pd.read_csv('data/processed/spy_test.csv')
        s_synth = pd.read_csv('data/synthetic/spy_synth.csv')
        
        train_and_save("spy", s_train, s_synth, s_test, "Target", train_evaluate_xgb_ts)
    except Exception as e:
        print(f"Error training Finance models: {e}")

if __name__ == '__main__':
    main()
