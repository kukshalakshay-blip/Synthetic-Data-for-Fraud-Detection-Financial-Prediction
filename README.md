# Synthetic Data for Fraud Detection & Financial Prediction

This project explores the generation and use of synthetic data in two key financial domains:
1. **Fraud Detection**: Using the Kaggle Credit Card Fraud Dataset.
2. **Financial Prediction**: Predicting stock movements or loan defaults.

The goal is to evaluate models trained on real data versus synthetic data across these two domains. We use synthetic data generators (such as SDV and CTGAN) to ensure data privacy while preserving the statistical distribution of the original datasets. Finally, the project is packaged into an interactive Streamlit application.

## Directory Structure
- `data/` : Datasets (raw, processed, synthetic).
- `notebooks/` : Experimental setups and exploratory data analysis.
- `src/` : Core python modules for processing, generation, and modeling.
- `app/` : Streamlit application code.

## Setup Instructions

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Collect the datasets and place them into `data/raw/` (e.g. `creditcard.csv` and `SPY_daily.csv` via the provided notebooks or scripts).

3. Execute the Offline Batch Pipeline:
Our project separates computationally heavy models into distinct scripts:
```bash
# Process raw datasets into tabular features
python src/01_process_data.py

# Synthesize new datasets using CTGAN (renders to data/synthetic/)
python src/02_generate_synthetic.py  

# Train XGBoost models on real vs synthetic sets and save to models/
python src/03_train_models.py
```

4. Launch the deployment demo dashboard:
```bash
streamlit run app/main.py
```
