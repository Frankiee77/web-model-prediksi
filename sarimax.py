# ==========================================================
# PREDIKSI DBD KOTA SUKABUMI
# MODEL SARIMAX
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from pathlib import Path

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Prediksi SARIMAX",
    page_icon="📈",
    layout="wide"
)

# ==========================================================
# PATH
# ==========================================================

MODEL_PATH = Path("models/sarimax_dbd.pkl")

DATASET_PATH = Path("dataset.xlsx")

EXOG_COLS = [
    "curah_hujan",
    "suhu",
    "kelembaban",
    "kepadatan"
]

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as f:

        model = pickle.load(f)

    return model


# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():

    df = pd.read_excel(DATASET_PATH)

    if "waktu" in df.columns:

        df["waktu"] = pd.to_datetime(df["waktu"])

    return df


# ==========================================================
# LOAD
# ==========================================================

try:

    model = load_model()

    df = load_dataset()

except Exception as e:

    st.error("Model gagal dimuat.")

    st.exception(e)

    st.stop()

# ==========================================================
# KATEGORI
# ==========================================================

def kategori(nilai):

    if nilai <= 20:
        return "🟢 Rendah"

    elif nilai <= 50:
        return "🟡 Sedang"

    elif nilai <= 100:
        return "🟠 Tinggi"

    else:
        return "🔴 Sangat Tinggi"

# ==========================================================
# HEADER
# ==========================================================

st.title("📈 Prediksi Kasus DBD Menggunakan SARIMAX")

st.markdown("""
Model **Seasonal AutoRegressive Integrated Moving Average with Exogenous Variables (SARIMAX)** digunakan untuk memprediksi jumlah kasus DBD berdasarkan data historis kasus dan variabel eksogen.
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Dataset",
        f"{len(df)} Data"
    )

with col2:

    st.metric(
        "Order",
        "(1,0,0)"
    )

with col3:

    st.metric(
        "Seasonal",
        "(2,0,0,12)"
    )

st.divider()

st.subheader("Input Variabel Bulan Berikutnya")

last = df.iloc[-1]

c1, c2 = st.columns(2)

with c1:
    curah = st.number_input(
        "Curah Hujan (mm)",
        value=float(last["curah_hujan"]),
        min_value=0.0
    )

    suhu = st.number_input(
        "Suhu (°C)",
        value=float(last["suhu"]),
        min_value=0.0,
        max_value=50.0
    )

with c2:
    kelembaban = st.number_input(
        "Kelembaban (%)",
        value=float(last["kelembaban"]),
        min_value=0.0,
        max_value=100.0
    )

    kepadatan = st.number_input(
        "Kepadatan",
        value=float(last["kepadatan"]),
        min_value=0.0
    )
  
st.divider()
predict = st.button(
    "🔮 Prediksi",
    use_container_width=True,
    type="primary"
)
st.write(type(model))
