import streamlit as st
from datetime import datetime
from typing import List

# Import utilitas
from utils.month_helper import get_previous_months, format_month_year
from utils.session import init_session_state, add_to_history
from utils.model_info import get_sarimax_model_info
from utils.plot import plot_sarimax_inputs
from utils.preprocess import compute_diff_exog

# Jika menggunakan statsmodels SARIMAX
try:
    import pickle
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# Inisialisasi session state untuk riwayat
init_session_state()

# Header halaman
st.title("Prediksi DBD dengan Model SARIMAX")

# Informasi Model
with st.expander("Informasi Model SARIMAX", expanded=True):
    model_info = get_sarimax_model_info()
    st.write(f"**Order:** {model_info['Order']}")
    st.write(f"**Seasonal Order:** {model_info['Seasonal Order']}")
    st.write(f"**Differencing Exogenous (orde):** {model_info['Differencing Exogenous']}")
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

# Tentukan periode t-1 (bulan sebelumnya)
prev_months = get_previous_months(year_pred, month_pred, n=1)
prev_year, prev_mon = prev_months[0]
prev_month_label = format_month_year(prev_year, prev_mon)
st.write(f"Data historis untuk masukan (periode t-1): {prev_month_label}")

# Input data eksogen untuk periode t-1 dan t
st.subheader("Masukkan Data Eksogen")
exog_vars = ["Curah Hujan", "Suhu", "Kelembaban", "Kepadatan Jentik"]
exog_values_prev = {}
exog_values_curr = {}

cols = st.columns(2 + len(exog_vars))
cols[0].write("Variabel")
cols[1].write(prev_month_label)
cols[2].write(month_name_pred)
for idx, var in enumerate(exog_vars, start=3):
    cols[idx].write(var)

for idx, var in enumerate(exog_vars):
    cols[0].write(var)
    exog_values_prev[var] = cols[1].number_input(f"{var}_t-1", min_value=0.0, step=0.1)
    exog_values_curr[var] = cols[2].number_input(f"{var}_t", min_value=0.0, step=0.1)

# Hitung differencing untuk setiap variabel
st.subheader("Ringkasan Differensial Eksogen")
diff_summary = {}
for var in exog_vars:
    diff = compute_diff_exog(exog_values_prev[var], exog_values_curr[var])
    diff_summary[f"Delta {var}"] = diff
st.write(diff_summary)

# Proses prediksi saat tombol ditekan
if st.button("Prediksi DBD"):
    # Siapkan data untuk model
    try:
        # Contoh: mengumpulkan differensial menjadi array input
        exog_diff = [diff_summary[f"Delta {var}"] for var in exog_vars]
        # Muat model SARIMAX (pastikan file model tersedia)
        if MODEL_AVAILABLE:
            with open("sarimax_model.pkl", "rb") as f:
                model = pickle.load(f)
            pred = model.predict(
                start=0, end=0, exog=[exog_diff]  # contoh metode; sesuaikan implementasi
            )
            prediction = float(pred.iloc[0])
        else:
            # Placeholder: menggunakan salah satu input
            prediction = float(exog_values_curr["Kepadatan Jentik"])
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        prediction = None

    if prediction is not None:
        kategori = "Rendah" if prediction < 50 else ("Sedang" if prediction < 100 else "Tinggi")
        st.subheader("Hasil Prediksi")
        st.write(f"**Bulan Prediksi:** {month_name_pred}")
        st.write(f"**Jumlah Kasus (diprediksi):** {prediction:.2f}")
        st.write(f"**Kategori:** {kategori}")

        # Tambahkan ke riwayat prediksi
        add_to_history("SARIMAX", month_name_pred, prediction)

        # Grafik berdasarkan input user (variabel eksogen)
        st.altair_chart(
            plot_sarimax_inputs(prev_month_label, month_name_pred,
                                [exog_values_prev["Curah Hujan"], exog_values_curr["Curah Hujan"]],
                                [exog_values_prev["Suhu"], exog_values_curr["Suhu"]],
                                [exog_values_prev["Kelembaban"], exog_values_curr["Kelembaban"]],
                                [exog_values_prev["Kepadatan Jentik"], exog_values_curr["Kepadatan Jentik"]]
                               ),
            use_container_width=True
        )

# Tampilkan riwayat prediksi jika ada
if st.session_state['history']:
    st.subheader("Riwayat Prediksi")
    import pandas as pd
    df_history = pd.DataFrame(st.session_state['history'])
    st.table(df_history)
