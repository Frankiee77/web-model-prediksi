import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path

from tensorflow.keras.models import load_model

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================

st.set_page_config(
    page_title="Prediksi DBD - LSTM",
    page_icon="🧠",
    layout="wide"
)

# ==========================================================
# PATH FILE
# ==========================================================

MODEL_PATH = Path("models/lstm_dbd.keras")
SCALER_FEATURE_PATH = Path("models/scaler_features.pkl")
SCALER_TARGET_PATH = Path("models/scaler_target.pkl")
DATASET_PATH = Path("dataset.xlsx")
WINDOW_LAG = 3
TARGET_COL = "kasus"
FEATURE_COLS = [
    "curah_hujan",
    "suhu",
    "kelembaban",
    "kepadatan"
]

ALL_COLS = [TARGET_COL] + FEATURE_COLS

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_lstm():
    model = load_model(
        MODEL_PATH,
        compile=False
    )
    return model

# ==========================================================
# LOAD SCALER
# ==========================================================

@st.cache_resource
def load_scaler():
    with open(SCALER_TARGET_PATH, "rb") as f:
        scaler_target = pickle.load(f)
    with open(SCALER_FEATURE_PATH, "rb") as f:
        scaler_features = pickle.load(f)
    return scaler_target, scaler_features

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
# LOAD RESOURCE
# ==========================================================

try:
    model = load_lstm()
    scaler_target, scaler_features = load_scaler()
    df = load_dataset()
except Exception as e:
    st.error("❌ Model tidak dapat dimuat.")
    st.exception(e)
    st.stop()

# ==========================================================
# PREPROCESS INPUT
# (IDENTIK DENGAN SAAT TRAINING)
# ==========================================================

def preprocess_input(df_input):
    # Normalisasi target
    target_scaled = scaler_target.transform(
        df_input[[TARGET_COL]]
    )
    # Normalisasi fitur
    feature_scaled = scaler_features.transform(
        df_input[FEATURE_COLS]
    )
    # Gabungkan kembali
    scaled = np.concatenate(
        [target_scaled, feature_scaled],
        axis=1
    )
    return scaled

# ==========================================================
# AMBIL 3 DATA TERAKHIR
# ==========================================================

def get_last_history():
    return (
        df[ALL_COLS]
        .tail(WINDOW_LAG)
        .reset_index(drop=True)
    )

# ==========================================================
# KATEGORI HASIL PREDIKSI
# ==========================================================

def kategori(prediksi):
    if prediksi <= 20:
        return "🟢 Rendah"
    elif prediksi <= 50:
        return "🟡 Sedang"
    elif prediksi <= 100:
        return "🟠 Tinggi"
    else:
        return "🔴 Sangat Tinggi"

# ==========================================================
# HEADER
# ==========================================================

st.title("🧠 Prediksi Kasus DBD Menggunakan LSTM")

st.markdown(
"""
Model **Long Short-Term Memory (LSTM)** digunakan untuk memprediksi
jumlah kasus Demam Berdarah Dengue (DBD) berdasarkan data historis
selama **3 bulan terakhir**.

Silakan masukkan data historis atau gunakan data terakhir yang tersedia
pada dataset.
"""
)
st.divider()

# ==========================================================
# INFORMASI MODEL
# ==========================================================

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Jumlah Dataset",
        f"{len(df)} Data"
    )
with col2:
    st.metric(
        "Window Lag",
        "3 Bulan"
    )
with col3:
    st.metric(
        "Jumlah Fitur",
        "5 Variabel"
    )
st.divider()

# ==========================================================
# AUTO FILL
# ==========================================================

st.subheader("Data Historis")
use_history = st.checkbox(
    "Gunakan 3 bulan terakhir dari dataset",
    value=True
)

if use_history:
    history = get_last_history()
else:
    history = pd.DataFrame(
        np.zeros((WINDOW_LAG, len(ALL_COLS))),
        columns=ALL_COLS
    )

# ==========================================================
# FORM INPUT DATA HISTORIS
# ==========================================================

st.markdown("## 📅 Data Historis 3 Bulan Terakhir")

st.info(
    "Masukkan data historis selama **3 bulan terakhir**. "
    "Data ini akan digunakan sebagai input model LSTM "
    "untuk memprediksi jumlah kasus DBD pada bulan berikutnya."
)

bulan_label = [
    "Bulan t-3",
    "Bulan t-2",
    "Bulan t-1"
]

input_history = []

for i in range(WINDOW_LAG):
    with st.container(border=True):
        st.markdown(f"### {bulan_label[i]}")
        c1, c2 = st.columns(2)
        with c1:
            kasus = st.number_input(
                "Kasus DBD",
                min_value=0.0,
                value=float(history.iloc[i]["kasus"]),
                step=1.0,
                key=f"kasus_{i}"
            )
            curah = st.number_input(
                "Curah Hujan (mm)",
                min_value=0.0,
                value=float(history.iloc[i]["curah_hujan"]),
                step=1.0,
                key=f"curah_{i}"
            )
            suhu = st.number_input(
                "Suhu (°C)",
                min_value=0.0,
                max_value=50.0,
                value=float(history.iloc[i]["suhu"]),
                step=0.1,
                key=f"suhu_{i}"
            )

        with c2:
            kelembaban = st.number_input(
                "Kelembaban (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(history.iloc[i]["kelembaban"]),
                step=0.1,
                key=f"kelembaban_{i}"
            )
            kepadatan = st.number_input(
                "Kepadatan Penduduk",
                min_value=0.0,
                value=float(history.iloc[i]["kepadatan"]),
                step=1.0,
                key=f"kepadatan_{i}"
            )

        input_history.append([
            kasus,
            curah,
            suhu,
            kelembaban,
            kepadatan
        ])
# ==========================================================
# DATAFRAME INPUT
# ==========================================================

input_df = pd.DataFrame(
    input_history,
    columns=ALL_COLS
)
st.divider()

st.subheader("Preview Data Input")
st.dataframe(
    input_df,
    use_container_width=True,
    hide_index=True
)
# ==========================================================
# TOMBOL PREDIKSI
# ==========================================================

st.divider()
left, center, right = st.columns([1,2,1])
with center:
    predict_btn = st.button(
        "🔮 Prediksi Kasus DBD Bulan Berikutnya",
        use_container_width=True,
        type="primary"
    )
# ==========================================================
# PROSES PREDIKSI
# ==========================================================

if predict_btn:
    try:
        # ---------------------------------------------
        # Normalisasi sesuai proses training
        # ---------------------------------------------
        scaled_input = preprocess_input(input_df)

        # ---------------------------------------------
        # Ubah shape menjadi
        # (1, window, fitur)
        # ---------------------------------------------
        X = scaled_input.reshape(
            1,
            WINDOW_LAG,
            len(ALL_COLS)
        )

        # ---------------------------------------------
        # Prediksi
        # ---------------------------------------------
        pred_scaled = model.predict(
            X,
            verbose=0
        )

        # ---------------------------------------------
        # Kembalikan ke skala asli
        # ---------------------------------------------
        pred = scaler_target.inverse_transform(
            pred_scaled
        )[0][0]
        pred = max(pred, 0)
        pred = round(pred)

        st.success("Prediksi berhasil dilakukan.")
        st.divider()
        st.subheader("📈 Hasil Prediksi")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                label="Prediksi Jumlah Kasus",
                value=f"{pred} Kasus"
            )
        
        with c2:
            st.metric(
                label="Kategori",
                value=kategori(pred)
            )
        st.divider()
        
        st.subheader("Ringkasan Input")
        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True
        )
        
        st.info(
        """
        **Keterangan**
        
        Prediksi dihasilkan menggunakan model **Long Short-Term Memory (LSTM)**
        dengan panjang window sebanyak **3 bulan**.
        
        Model dibangun menggunakan:
        
        - 5 variabel
        - Window Lag = 3
        - LSTM Units = 8
        - Optimizer Adam
        - Huber Loss
        """
        )
    except Exception as e:
        st.error("Prediksi gagal.")
        st.exception(e)
        st.stop()
# ==========================================================
# HASIL PREDIKSI
# ==========================================================

