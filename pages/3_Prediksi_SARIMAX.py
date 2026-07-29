import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.special import inv_boxcox

# Import utilitas
from utils.month_helper import get_previous_months, format_month_year
from utils.session import init_session_state, add_to_history
from utils.model_info import get_sarimax_model_info
from utils.plot import plot_sarimax_inputs
from utils.loader import load_sarimax, load_dataset

# ==========================================================
# LOAD MODEL & DATASET
# ==========================================================

artifacts      = load_sarimax()
result         = artifacts["model"]
fitted_lambda  = artifacts["fitted_lambda"]
# last_values diambil dari dataset terbaru (bukan dari artifacts)
# agar referensi differencing selalu mengacu pada data paling akhir
df             = load_dataset()
last_row       = df.iloc[-1]
last_values    = {
    "curah_hujan": float(last_row["curah_hujan"]),
    "suhu":        float(last_row["suhu"]),
    "kelembaban":  float(last_row["kelembaban"]),
    "kepadatan":   float(last_row["kepadatan"])
}

# Inisialisasi session state untuk riwayat
init_session_state()

# ==========================================================
# KONSTANTA
# ==========================================================

# Variabel eksogen yang digunakan model (3 variabel)
# Kepadatan dikeluarkan karena tidak stasioner setelah berbagai transformasi
EXOG_VARS = ["curah_hujan", "suhu", "kelembaban", "kepadatan"]

LABEL_VARS = {
    "curah_hujan": "Curah Hujan (mm)",
    "suhu":        "Suhu (°C)",
    "kelembaban":  "Kelembaban (%)",
    "kepadatan":   "Kepadatan (jiwa/km2)"
}

# ==========================================================
# FUNGSI BANTU
# ==========================================================

def hitung_exog_transformed(input_user: dict) -> np.ndarray:
    """
    Transformasi input sebelum masuk model SARIMAX.

    - curah_hujan : differencing (nilai_input - nilai_bulan_terakhir_dataset)
    - suhu        : differencing (nilai_input - nilai_bulan_terakhir_dataset)
    - kelembaban  : nilai asli (sudah stasioner, tidak di-diff)
    - kepadatan   : differencing (nilai_input - nilai_bulan_terakhir_dataset)

    Urutan kolom harus identik dengan saat training:
    [curah_hujan_diff, suhu_diff, kelembaban_nodiff]
    """
    curah_diff = input_user["curah_hujan"] - last_values["curah_hujan"]
    suhu_diff  = input_user["suhu"]        - last_values["suhu"]
    kelembaban = input_user["kelembaban"]
    kepadatan_diff  = input_user["kepadatan"]   - last_values["kepadatan"]

    return np.array([[curah_diff, suhu_diff, kelembaban, kepadatan]])

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

st.title("Prediksi DBD dengan Model SARIMAX")

with st.expander("ℹ️ Informasi Model SARIMAX", expanded=True):
    model_info = get_sarimax_model_info()
    st.write(f"**Order:** {model_info['Order']}")
    st.write(f"**Seasonal Order:** {model_info['Seasonal Order']}")
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

# Periode t-1: satu bulan sebelum bulan prediksi (untuk referensi user)
prev_months     = get_previous_months(year_pred, month_pred, n=1)
prev_year, prev_mon = prev_months[0]
prev_label      = format_month_year(prev_year, prev_mon)

st.divider()

# ==========================================================
# INPUT VARIABEL EKSOGEN
# ==========================================================

st.subheader("Masukkan Data Variabel Iklim")
st.info(
    "Masukkan nilai variabel iklim pada **bulan yang ingin diprediksi**. "
    )

# Tampilkan nilai referensi untuk transparansi
with st.expander("📋 Nilai referensi bulan terakhir dataset", expanded=False):
    ref_df = pd.DataFrame([{
        LABEL_VARS["curah_hujan"]: last_values["curah_hujan"],
        LABEL_VARS["suhu"]:        last_values["suhu"],
        LABEL_VARS["kelembaban"]:  last_values["kelembaban"],
        LABEL_VARS["kepadatan"]:  last_values["kepadatan"]
    }])
    st.dataframe(ref_df, use_container_width=True)
    st.caption(
        "Differencing = Nilai Input − Nilai Referensi. "
        "Nilai referensi diambil dari baris terakhir dataset secara otomatis."
    )

col1, col2 = st.columns(2)
with col1:
    curah = st.number_input(
        LABEL_VARS["curah_hujan"],
        min_value=0.0, step=1.0,
        value=last_values["curah_hujan"],
        help="Curah hujan bulan yang ingin diprediksi (mm)"
    )
    suhu = st.number_input(
        LABEL_VARS["suhu"],
        min_value=0.0, max_value=50.0, step=0.1,
        value=last_values["suhu"],
        help="Suhu rata-rata bulan yang ingin diprediksi (°C)"
    )
with col2:
    kelembaban = st.number_input(
        LABEL_VARS["kelembaban"],
        min_value=0.0, max_value=100.0, step=0.1,
        value=last_values["kelembaban"],
        help="Kelembaban rata-rata bulan yang ingin diprediksi (%)"
    )
    kepadatan = st.number_input(
        LABEL_VARS["kepadatan"],
        min_value=0.0, max_value=10000.0, step=10.0,
        value=last_values["kepadatan"],
        help="Kepadatan Penduduk bulan yang ingin diprediksi (jiwa/km2)"
    )

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
        input_user = {
            "curah_hujan": curah,
            "suhu":        suhu,
            "kelembaban":  kelembaban,
            "kepadatan":   kepadatan
        }

        # 1. Hitung differencing otomatis
        exog_transformed = hitung_exog_transformed(input_user)
        # shape (1, 3): [curah_diff, suhu_diff, kelembaban_asli, kepadatan_diff]

        # 2. Forecast dalam skala Box-Cox
        pred_transformed = result.forecast(steps=1, exog=exog_transformed)[0]

        # 3. Inverse Box-Cox ke skala kasus asli
        pred = float(inv_boxcox(pred_transformed, fitted_lambda))
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

        # 4. Detail transformasi (untuk transparansi & keperluan demo)
        with st.expander("🔍 Detail transformasi input", expanded=False):
            detail_df = pd.DataFrame({
                "Variabel":          ["Curah Hujan", "Suhu", "Kelembaban", "Kepadatan"],
                "Nilai Input":       [curah, suhu, kelembaban, kepadatan],
                "Nilai Referensi":   [
                    last_values["curah_hujan"],
                    last_values["suhu"],
                    last_values["kelembaban"],
                    last_values["kepadatan"]
                ],
                "Nilai ke Model":    [
                    round(exog_transformed[0][0], 4),
                    round(exog_transformed[0][1], 4),
                    round(exog_transformed[0][2], 4),
                    round(exog_transformed[0][3], 4)
                ],
                "Transformasi":      ["Differencing", "Differencing", "Nilai Asli", "Differencing"],
            })
            st.dataframe(detail_df, use_container_width=True)
            st.caption(
                f"Prediksi dalam skala Box-Cox (λ={fitted_lambda:.4f}) "
                f"= {pred_transformed:.4f} → skala kasus asli = {pred}."
            )

        # 5. Tambah ke riwayat
        add_to_history("SARIMAX", month_name_pred, pred)

        # 6. Grafik variabel eksogen
        st.divider()
        st.subheader("📊 Visualisasi Input Variabel Iklim")
        st.altair_chart(
            plot_sarimax_inputs(
                prev_label, month_name_pred,
                [last_values["curah_hujan"], curah],
                [last_values["suhu"],        suhu],
                [last_values["kelembaban"],  kelembaban],
                [last_values["kepadatan"],  kepadatan],
            ),
            use_container_width=True
        )
        st.caption(
            "Grafik menampilkan perubahan variabel iklim dari bulan referensi "
            "ke bulan prediksi. Differencing dihitung dari selisih kedua nilai tersebut."
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
