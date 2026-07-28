import streamlit as st
from datetime import datetime
from typing import List

# Import utilitas
from utils.month_helper import get_previous_months, format_month_year
from utils.session import init_session_state, add_to_history
from utils.model_info import get_lstm_model_info
from utils.plot import plot_lstm_prediction
from utils.preprocess import prepare_lstm_input
from utils.loader import (
    load_lstm,
    load_lstm_scaler
)

model = load_lstm()

scaler_target, scaler_feature = load_lstm_scaler()

# Inisialisasi session state untuk riwayat
init_session_state()

# Header halaman
st.title("Prediksi DBD dengan Model LSTM")

# Informasi Model (dapat ditempatkan di sidebar atau atas)
with st.expander("Informasi Model LSTM", expanded=True):
    model_info = get_lstm_model_info()
    st.write(f"**Model:** {model_info['Model']}")
    st.write(f"**Window:** {model_info['Window']} bulan")
    st.write(f"**Jumlah Variabel Input:** {model_info['Jumlah Variabel']}")
    st.write(f"**Metode Validasi:** {model_info['Validasi']}")
    st.write(f"**MAE:** {model_info['MAE']}")
    st.write(f"**RMSE:** {model_info['RMSE']}")
    st.write(f"**MAPE:** {model_info['MAPE']}%")

# Pemilihan bulan prediksi
st.header("Pilih Bulan Prediksi")
col_year, col_month = st.columns(2)
with col_year:
    year_pred = st.number_input("Tahun Prediksi", min_value=2021, max_value=2030, value=datetime.now().year, step=1)
with col_month:
    month_pred = st.selectbox("Bulan Prediksi", [i for i in range(1, 13)])
month_name_pred = format_month_year(year_pred, month_pred)
st.write(f"Prediksi: **{month_name_pred}**")

# Tentukan 3 bulan historis sebelumnya
prev_months = get_previous_months(year_pred, month_pred, n=3)
# Dapat ditampilkan
hist_labels = [format_month_year(y, m) for (y, m) in prev_months]
st.write("Data historis untuk masukan (3 bulan terakhir):", ", ".join(hist_labels))

# Input data untuk 3 bulan historis
st.subheader("Masukkan Data Historis")
variables = ["Kasus", "Curah Hujan", "Suhu", "Kelembaban", "Kepadatan Jentik"]
input_values = {var: [] for var in variables}

# Buat kolom untuk tabel input
cols = st.columns(len(variables) + 1)
cols[0].write("Bulan")
for idx, var in enumerate(variables, start=1):
    cols[idx].write(var)

for idx, (y, m) in enumerate(prev_months):
    month_label = format_month_year(y, m)
    cols[0].write(month_label)
    input_values["Kasus"].append(cols[1].number_input(f"Kasus_{y}_{m}", min_value=0, step=1))
    input_values["Curah Hujan"].append(cols[2].number_input(f"CurahHujan_{y}_{m}", min_value=0.0, step=0.1))
    input_values["Suhu"].append(cols[3].number_input(f"Suhu_{y}_{m}", min_value=0.0, step=0.1))
    input_values["Kelembaban"].append(cols[4].number_input(f"Kelembaban_{y}_{m}", min_value=0.0, step=0.1))
    input_values["Kepadatan Jentik"].append(cols[5].number_input(f"Kepadatan_{y}_{m}", min_value=0.0, step=0.1))

# Ringkasan input oleh user
st.subheader("Ringkasan Input")
import pandas as pd
df_input = pd.DataFrame(input_values, index=hist_labels)
st.table(df_input)

# Proses prediksi saat tombol ditekan
if st.button("Prediksi DBD"):
    # Siapkan data untuk model
    # Contoh transformasi: mengumpulkan input ke dalam format yang dibutuhkan model
    try:
        import numpy as np
        input_array = np.array(list(zip(
            input_values["Kasus"],
            input_values["Curah Hujan"],
            input_values["Suhu"],
            input_values["Kelembaban"],
            input_values["Kepadatan Jentik"]
        )), dtype=float)
        # Jika diperlukan, ubah bentuk array (misal untuk LSTM) dan normalisasi:
        # X_input = prepare_lstm_input(input_array)
        X_input = input_array.reshape(1, input_array.shape[0], input_array.shape[1])
        # Muat model LSTM (pastikan file model tersedia)
        if MODEL_AVAILABLE:
            model = load_model("lstm_model.h5")
            pred = model.predict(X_input)
            # Jika menggunakan scaler untuk invers transform, lakukan di sini
            prediction = float(pred.flatten()[0])
        else:
            prediction = float(input_values["Kasus"][-1])  # placeholder fallback
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        prediction = None

    if prediction is not None:
        # Tentukan kategori (contoh sederhana)
        kategori = "Rendah" if prediction < 50 else ("Sedang" if prediction < 100 else "Tinggi")
        st.subheader("Hasil Prediksi")
        st.write(f"**Bulan Prediksi:** {month_name_pred}")
        st.write(f"**Jumlah Kasus (diprediksi):** {prediction:.2f}")
        st.write(f"**Kategori:** {kategori}")

        # Tambahkan ke riwayat prediksi
        add_to_history("LSTM", month_name_pred, prediction)

        # Grafik berdasarkan input user dan prediksi
        st.altair_chart(plot_lstm_prediction(hist_labels, list(input_values["Kasus"]), prediction), use_container_width=True)

# Tampilkan riwayat prediksi jika ada
if st.session_state['history']:
    st.subheader("Riwayat Prediksi")
    df_history = pd.DataFrame(st.session_state['history'])
    st.table(df_history)
