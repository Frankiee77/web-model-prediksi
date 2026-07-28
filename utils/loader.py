from pathlib import Path
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

MODEL_DIR = Path("models")

# ---------------- LSTM ----------------

@st.cache_resource
def load_lstm():

    model = load_model(
        MODEL_DIR / "models/lstm_dbd.keras",
        compile=False
    )

    return model


@st.cache_resource
def load_lstm_scaler():

    with open(MODEL_DIR / "scaler_target.pkl", "rb") as f:
        scaler_target = pickle.load(f)

    with open(MODEL_DIR / "scaler_features.pkl", "rb") as f:
        scaler_feature = pickle.load(f)

    return scaler_target, scaler_feature


# ---------------- SARIMAX ----------------

@st.cache_resource
def load_sarimax():

    with open("models/sarimax_artifacts.pkl","rb") as f:

        artifacts = {

            "model": result_sarimax,
        
            "fitted_lambda": fitted_lambda,
        
            "last_values": {
                'curah_hujan': 2209,
                'suhu': 24.01,
                'kelembaban': 89.87,
                'kepadatan': 7820
            },
        
            "order": ORDER,
        
            "seasonal_order": SEASONAL_ORDER,
        
            "exog_cols": exog_cols,
        
            "mae": mae_sarimax,
        
            "mape": mape_sarimax
        
        }

    return artifacts
