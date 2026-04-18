import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_fraud_data(filepath='data/raw/creditcard.csv', test_size=0.2, random_state=42, sample_frac=1.0):
    """Loads and preprocesses the Kaggle credit card fraud dataset.
    sample_frac: Float between 0 and 1 to quickly sample a smaller subset for faster runs.
    """
    df = pd.read_csv(filepath)
    
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
    
    # Scale 'Amount' and 'Time'
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    
    df['Amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test, df

def load_and_preprocess_financial_data(filepath='data/raw/SPY_daily.csv', test_size=0.2, random_state=42):
    """Loads and preprocesses the SPY stock market dataset."""
    df = pd.read_csv(filepath, skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Calculate Daily Return
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Moving averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_15'] = df['Close'].rolling(window=15).mean()
    
    # Add lag features
    for lag in range(1, 4):
        df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)
    
    # Create target: 1 if Next Day Close > Today's Close
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df = df.dropna()
    
    # Use features independent of absolute price for better synthetic gen
    features = ['Daily_Return', 'SMA_5', 'SMA_15', 'Volume', 'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3']
    
    # Standardize
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Keep final dataset tabular
    final_df = df[features + ['Target']].copy()
    
    X = final_df.drop('Target', axis=1)
    y = final_df['Target']
    
    # Time-series aware split (no random shuffling for testing, chronologically split)
    split_idx = int(len(final_df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test, final_df
