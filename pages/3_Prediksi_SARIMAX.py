import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
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
            st.divider()

            st.subheader("📈 Visualisasi Prediksi")
            
            fig = plot_prediction(df, pred)
            
            st.pyplot(fig)
            
            st.caption(
                "Grafik menampilkan 12 bulan terakhir data historis "
                "dan hasil prediksi bulan berikutnya menggunakan model SARIMAX."
            )

    except Exception as e:

        st.error("Forecast gagal.")

        st.exception(e)
        
def plot_prediction(df, pred):

    history = df.tail(12).copy()

    history = history.reset_index(drop=True)

    x_hist = list(range(len(history)))

    x_pred = len(history)

    fig, ax = plt.subplots(figsize=(11,4))

    # ===========================
    # DATA HISTORIS
    # ===========================

    ax.plot(
        x_hist,
        history["kasus"],
        marker="o",
        linewidth=2,
        label="Data Historis"
    )

    # ===========================
    # GARIS MENUJU PREDIKSI
    # ===========================

    ax.plot(
        [x_hist[-1], x_pred],
        [history["kasus"].iloc[-1], pred],
        "--",
        linewidth=2,
        color="red"
    )

    # ===========================
    # TITIK PREDIKSI
    # ===========================

    ax.scatter(
        x_pred,
        pred,
        color="red",
        s=120,
        label="Prediksi"
    )

    # ===========================
    # LABEL NILAI
    # ===========================

    for i, y in enumerate(history["kasus"]):

        ax.text(
            i,
            y+3,
            f"{int(y)}",
            fontsize=8,
            ha="center"
        )

    ax.text(
        x_pred,
        pred+3,
        f"{int(pred)}",
        fontsize=9,
        color="red",
        ha="center",
        fontweight="bold"
    )

    # ===========================
    # LABEL X
    # ===========================

    labels = []

    if "waktu" in df.columns:

        labels = history["waktu"].dt.strftime("%b\n%Y").tolist()

        next_month = (
            history["waktu"].iloc[-1]
            + pd.DateOffset(months=1)
        ).strftime("%b\n%Y")

        labels.append(next_month)

    else:

        labels = [str(i+1) for i in range(len(history))]
        labels.append("Pred")

    ax.set_xticks(list(range(len(labels))))

    ax.set_xticklabels(labels)

    ax.set_ylabel("Kasus DBD")

    ax.set_xlabel("Periode")

    ax.set_title("Visualisasi Data Historis dan Prediksi")

    ax.grid(alpha=0.3)

    ax.legend()

    plt.tight_layout()

    return fig
    
