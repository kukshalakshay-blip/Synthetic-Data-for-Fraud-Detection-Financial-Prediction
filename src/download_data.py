import os
import shutil
import kagglehub
import yfinance as yf

def main():
    # Ensure data directory exists
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # 1. Kaggle setup
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_87bd1626293cb5123e491dffa841186d'
    print("Downloading Kaggle Credit Card Fraud dataset...")
    try:
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        dest_file = 'data/raw/creditcard.csv'
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith('.csv'):
                    shutil.copy(os.path.join(path, f), dest_file)
                    print(f"Copied {f} to {dest_file}")
        else:
            shutil.copy(path, dest_file)
            print(f"Copied dataset to {dest_file}")
    except Exception as e:
        print("Error downloading from Kaggle:", e)

    # 2. Yahoo Finance setup
    print("\nDownloading Yahoo Finance SPY data...")
    try:
        df = yf.download('SPY', start='2015-01-01', end='2023-01-01')
        df.to_csv('data/raw/SPY_daily.csv')
        print("Downloaded SPY_daily.csv")
    except Exception as e:
        print("Error downloading from Yahoo Finance:", e)

if __name__ == '__main__':
    main()
