import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# Import utilitas
from utils.month_helper import get_previous_months, format_month_year
from utils.session import init_session_state, add_to_history
from utils.model_info import get_lstm_model_info
from utils.plot import plot_lstm_prediction
from utils.loader import load_lstm, load_lstm_scaler

# ==========================================================
# LOAD MODEL & SCALER
# ==========================================================

model                          = load_lstm()
scaler_target, scaler_features = load_lstm_scaler()

# Inisialisasi session state untuk riwayat
init_session_state()

# ==========================================================
# KONSTANTA
# ==========================================================

WINDOW_LAG   = 3
TARGET_COL   = "kasus"
# Urutan fitur harus identik dengan saat training
# Kepadatan tetap dimasukkan sebagai input LSTM karena
# LSTM tidak mensyaratkan stasioneritas variabel input
FEATURE_COLS = ["curah_hujan", "suhu", "kelembaban", "kepadatan"]
ALL_COLS     = [TARGET_COL] + FEATURE_COLS   # 5 kolom total

LABEL_VARS = {
    "kasus":       "Kasus DBD",
    "curah_hujan": "Curah Hujan (mm)",
    "suhu":        "Suhu (°C)",
    "kelembaban":  "Kelembaban (%)",
    "kepadatan":   "Kepadatan Penduduk (jiwa/km²)",
}

# ==========================================================
# FUNGSI BANTU
# ==========================================================

def preprocess_window(input_rows: list) -> np.ndarray:
    """
    Normalisasi window 3 baris x 5 kolom identik dengan proses training.
    Mengembalikan array shape (1, 3, 5) siap masuk model LSTM.
    """
    arr = np.array(input_rows, dtype=float)                        # (3, 5)
    target_scaled  = scaler_target.transform(arr[:, [0]])          # (3, 1)
    feature_scaled = scaler_features.transform(arr[:, 1:])         # (3, 4)
    scaled = np.concatenate([target_scaled, feature_scaled], axis=1)  # (3, 5)
    return scaled.reshape(1, WINDOW_LAG, len(ALL_COLS))            # (1, 3, 5)

def kategori(nilai: float) -> str:
    if nilai <= 20:
        return "🟢 Rendah"
    elif nilai <= 50:
        return "🟡 Sedang"
    elif nilai <= 100:
        return "🟠 Tinggi"
    return "🔴 Sangat Tinggi"

# ==========================================================
# HEADER & INFORMASI MODEL
# ==========================================================

st.title("Prediksi DBD dengan Model LSTM")

with st.expander("ℹ️ Informasi Model LSTM", expanded=True):
    model_info = get_lstm_model_info()
    c1 = st.columns(1)
    with c1:
        st.write(f"**Model:** {model_info['Model']}")
        st.write(f"**Window Input:** {model_info['Window']} bulan")
        st.write(f"**Jumlah Variabel:** {model_info['Jumlah Variabel']}")

st.divider()

# ==========================================================
# PEMILIHAN BULAN PREDIKSI
# ==========================================================

st.header("Pilih Bulan Prediksi")
col_year, col_month = st.columns(2)
with col_year:
    year_pred = st.number_input(
        "Tahun Prediksi",
        min_value=2021, max_value=2030,
        value=datetime.now().year, step=1
    )
with col_month:
    month_pred = st.selectbox(
        "Bulan Prediksi",
        list(range(1, 13)),
        format_func=lambda m: datetime(2000, m, 1).strftime("%B")
    )

month_name_pred = format_month_year(year_pred, month_pred)
st.write(f"Prediksi untuk: **{month_name_pred}**")

prev_months = get_previous_months(year_pred, month_pred, n=WINDOW_LAG)
hist_labels = [format_month_year(y, m) for (y, m) in prev_months]
st.caption(
    "Data historis yang dibutuhkan: "
    + ", ".join(f"**{l}**" for l in hist_labels)
)

st.divider()

# ==========================================================
# INPUT DATA HISTORIS 3 BULAN
# ==========================================================

st.subheader("Masukkan Data Historis 3 Bulan Terakhir")
st.info(
    "Masukkan data aktual untuk setiap variabel pada 3 bulan sebelum "
    "bulan yang ingin diprediksi. Data ini digunakan sebagai window "
    "input model LSTM sesuai metode Walk-Forward Validation."
)

input_rows = []

for i, (y, m) in enumerate(prev_months):
    label = hist_labels[i]
    with st.container(border=True):
        st.markdown(f"### {label} &nbsp;_(t-{WINDOW_LAG - i})_")
        c1, c2 = st.columns(2)
        with c1:
            kasus = st.number_input(
                LABEL_VARS["kasus"], min_value=0, step=1,
                key=f"kasus_{y}_{m}"
            )
            curah = st.number_input(
                LABEL_VARS["curah_hujan"], min_value=0.0, step=1.0,
                key=f"curah_{y}_{m}"
            )
            suhu = st.number_input(
                LABEL_VARS["suhu"], min_value=0.0, max_value=50.0, step=0.1,
                key=f"suhu_{y}_{m}"
            )
        with c2:
            kelembaban = st.number_input(
                LABEL_VARS["kelembaban"], min_value=0.0, max_value=100.0, step=0.1,
                key=f"kelembaban_{y}_{m}"
            )
            kepadatan = st.number_input(
                LABEL_VARS["kepadatan"], min_value=0.0, step=1.0,
                key=f"kepadatan_{y}_{m}"
            )
        input_rows.append([kasus, curah, suhu, kelembaban, kepadatan])

# Tabel ringkasan
st.subheader("Ringkasan Input")
df_input = pd.DataFrame(
    input_rows,
    index=hist_labels,
    columns=[LABEL_VARS[c] for c in ALL_COLS]
)
st.dataframe(df_input, use_container_width=True)

st.divider()

# ==========================================================
# TOMBOL PREDIKSI
# ==========================================================

left, center, right = st.columns([1, 2, 1])
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
        # 1. Normalisasi window input
        X_input = preprocess_window(input_rows)          # (1, 3, 5)

        # 2. Prediksi
        pred_scaled = model.predict(X_input, verbose=0)  # (1, 1)

        # 3. Inverse transform ke skala kasus asli
        pred = float(scaler_target.inverse_transform(pred_scaled)[0][0])
        pred = round(max(pred, 0))

        st.success("Prediksi berhasil dilakukan.")
        st.divider()
        st.subheader("📈 Hasil Prediksi")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Bulan Prediksi", month_name_pred)
        with c2:
            st.metric("Prediksi Jumlah Kasus", f"{pred} Kasus")
        with c3:
            st.metric("Kategori Risiko", kategori(pred))

        # 4. Tambah ke riwayat
        add_to_history("LSTM", month_name_pred, pred)

        # 5. Grafik
        st.divider()
        st.subheader("📊 Visualisasi Prediksi")
        kasus_hist = [row[0] for row in input_rows]
        st.altair_chart(
            plot_lstm_prediction(hist_labels, kasus_hist, pred),
            use_container_width=True
        )
        st.caption(
            "Grafik menampilkan data historis kasus DBD 3 bulan terakhir "
            "dan hasil prediksi bulan berikutnya menggunakan model LSTM "
            "dengan metode Walk-Forward Validation."
        )

    except Exception as e:
        st.error("Prediksi gagal.")
        st.exception(e)

# ==========================================================
# RIWAYAT PREDIKSI
# ==========================================================

if st.session_state.get("history"):
    st.divider()
    st.subheader("📋 Riwayat Prediksi")
    df_history = pd.DataFrame(st.session_state["history"])
    st.dataframe(df_history, use_container_width=True)
