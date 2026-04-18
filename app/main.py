import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Synthetic Base", layout="wide", page_icon="🧬", initial_sidebar_state="expanded")

# Custom CSS for premium UI
st.markdown("""
<style>
    h1 {
        background: linear-gradient(90deg, #A18CD1 0%, #FBC2EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        background: rgba(255,255,255, 0.05);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #A18CD1;
        transition: transform 0.2s ease;
    }
    .stMetric:hover {
        transform: translateY(-5px);
    }
</style>
""", unsafe_allow_html=True)

st.title("Synthetic Data Studio & ML Engine")
st.markdown("Build, train, and compare your standard real financial data vs heavily secure generated synthetic data. *(Loading from Offline Pipeline Storage)*")

# Sidebar Configuration
st.sidebar.title("Configuration")
domain = st.sidebar.radio("Select Domain", ["Fraud Detection (Credit Cards)", "Financial Prediction (SPY)"])

# Setup internal mapping based on domain
if domain == "Fraud Detection (Credit Cards)":
    prefix = "fraud"
    target_col = "Class"
    raw_path = "data/raw/creditcard.csv"
else:
    prefix = "spy"
    target_col = "Target"
    raw_path = "data/raw/SPY_daily.csv"

@st.cache_data
def load_raw():
    return pd.read_csv(raw_path)

@st.cache_data
def load_processed(p):
    test_df = pd.read_csv(f"data/processed/{p}_test.csv")
    synth_df = pd.read_csv(f"data/synthetic/{p}_synth.csv")
    return test_df, synth_df

# UI logic
tabs = st.tabs(["📊 Exploration", "🧬 Synthetic View", "🤖 Fast Model Inference"])

try:
    with tabs[0]:
        st.subheader(f"Raw Dataset Exploration - {domain}")
        raw_df = load_raw()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Real Row Count", len(raw_df))
            st.metric("Total Features", len(raw_df.columns))
            # Just count the unique values roughly without throwing errors if the domain misses target occasionally in raw CSV parsing
            if target_col in raw_df.columns:
                st.metric("Target Positive Count", len(raw_df[raw_df[target_col] == 1]))
            
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            if target_col in raw_df.columns:
                sns.countplot(data=raw_df, x=target_col, ax=ax, palette='magma')
                if prefix == "fraud":
                    plt.yscale('log')
                plt.title(f'Distribution of {target_col}')
            st.pyplot(fig, use_container_width=True)
            
        st.dataframe(raw_df.head(15), use_container_width=True)

    with tabs[1]:
        st.subheader("Offline Synthetic Generation Status")
        st.markdown("Instead of generating randomly generated parameters inline asynchronously, this loads the dedicated CTGAN serialized pipeline components from `data/synthetic`.")
        
        test_df, synth_df = load_processed(prefix)
        st.success(f"Loaded {len(synth_df)} synthesized offline samples globally.")
        
        st.metric("Synthetic Generation Rows", len(synth_df))
        st.dataframe(synth_df.head(15), use_container_width=True)

    with tabs[2]:
        st.subheader("Evaluate XGBoost Loaded Trees")
        st.markdown("We'll evaluate offline compiled standard XGBoost Classifiers (both real and synthetic architectures).")
        
        if st.button("Load and Evaluate Disk Parameters"):
            test_df, _ = load_processed(prefix)
            X_test, y_test = test_df.drop(target_col, axis=1), test_df[target_col]
            
            with st.spinner("Compiling joblib structures..."):
                model_real = joblib.load(f"models/{prefix}_xgb_real.joblib")
                model_synth = joblib.load(f"models/{prefix}_xgb_synth.joblib")
                
                preds_real = model_real.predict(X_test)
                preds_synth = model_synth.predict(X_test)
                
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Loaded Original Model")
                st.info("Trained exclusively on `data/processed/*_train.csv`")
                st.metric("Base Accuracy Rate", f"{accuracy_score(y_test, preds_real):.4f}")

            with col2:
                st.markdown("### Loaded Synthetic Model")
                st.info("Trained exclusively on `data/synthetic/*_synth.csv`")
                st.metric("Synthetic Accuracy Rate", f"{accuracy_score(y_test, preds_synth):.4f}")

except Exception as e:
    st.error(f"Waiting for offline execution (Run the src/ Python scripts first!): {e}")
