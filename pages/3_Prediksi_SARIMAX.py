import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path

st.set_page_config(
    page_title="Prediksi DBD - SARIMAX",
    page_icon="📈",
    layout="wide"
)

DATASET_PATH = Path("dataset.xlsx")

EXOG_COLS = [
    "curah_hujan",
    "suhu",
    "kelembaban",
    "kepadatan"
]

ORDER = (1, 0, 0)
SEASONAL_ORDER = (2, 0, 0, 12)

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():
    return pd.read_excel(DATASET_PATH)

# ==========================================================
# FIT SARIMAX
# ==========================================================

@st.cache_resource
def fit_sarimax(df):

    model = SARIMAX(
        endog=df["kasus"],
        exog=df[EXOG_COLS],
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(
        disp=False,
        maxiter=150
    )
    return result

# ==========================================================
# LOAD RESOURCE
# ==========================================================

try:
    df = load_dataset()
    with st.spinner("Membangun model SARIMAX..."):
        result = fit_sarimax(df)
    
except Exception as e:
    st.error("Model gagal dibuat.")
    st.exception(e)
    st.stop()
    
def kategori(nilai):
    if nilai <= 20:
        return "🟢 Rendah"
    elif nilai <= 50:
        return "🟡 Sedang"
    elif nilai <= 100:
        return "🟠 Tinggi"
    return "🔴 Sangat Tinggi"

st.title("📈 Prediksi Kasus DBD Menggunakan SARIMAX")

st.markdown("""
Model **Seasonal AutoRegressive Integrated Moving Average with Exogenous Variables (SARIMAX)** digunakan untuk memprediksi jumlah kasus DBD satu bulan ke depan berdasarkan data historis serta variabel eksogen.
""")

st.divider()

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Jumlah Dataset", len(df))
with c2:
    st.metric("Order", "(1,0,0)")
with c3:
    st.metric("Seasonal Order", "(2,0,0,12)")
    
st.divider()

st.subheader("Input Variabel Bulan Berikutnya")

last = df.iloc[-1]
col1, col2 = st.columns(2)
with col1:
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

with col2:
    kelembaban = st.number_input(
        "Kelembaban (%)",
        value=float(last["kelembaban"]),
        min_value=0.0,
        max_value=100.0
    )
    kepadatan = st.number_input(
        "Kepadatan Penduduk",
        value=float(last["kepadatan"]),
        min_value=0.0
    )
    
st.divider()

predict = st.button(
    "🔮 Prediksi",
    use_container_width=True,
    type="primary"
)

if predict:

    try:
        exog_next = pd.DataFrame(
            [[curah, suhu, kelembaban, kepadatan]],
            columns=EXOG_COLS
        )
        pred = result.forecast(
            steps=1,
            exog=exog_next
        )
        pred = round(max(float(pred.iloc[0]),0))

        st.success("Prediksi berhasil dilakukan.")

        st.divider()

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Prediksi Jumlah Kasus",
                f"{pred} Kasus"
            )

        with c2:
            st.metric(
                "Kategori",
                kategori(pred)
            )

    except Exception as e:

        st.error("Forecast gagal.")

        st.exception(e)
