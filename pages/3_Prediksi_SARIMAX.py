import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
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

artifacts     = load_sarimax()
result        = artifacts["model"]
fitted_lambda = artifacts["fitted_lambda"]

df       = load_dataset()
last_row = df.iloc[-1]
last_values = {
    "curah_hujan": float(last_row["curah_hujan"]),
    "suhu":        float(last_row["suhu"]),
    "kelembaban":  float(last_row["kelembaban"]),
    "kepadatan":   float(last_row["kepadatan"]),
}

init_session_state()

# ==========================================================
# KONSTANTA
# ==========================================================

# Urutan kolom identik dengan saat training:
# [curah_hujan_diff, suhu_diff, kelembaban_nodiff, kepadatan_diff]
EXOG_COLS = ["curah_hujan", "suhu", "kelembaban", "kepadatan"]

LABEL_VARS = {
    "curah_hujan": "Curah Hujan (mm)",
    "suhu":        "Suhu (°C)",
    "kelembaban":  "Kelembaban (%)",
    "kepadatan":   "Kepadatan Penduduk (jiwa/km²)",
}

# Variabel yang di-differencing (t - t-1)
# Kelembaban tidak di-diff karena sudah stasioner
DIFF_VARS = ["curah_hujan", "suhu", "kepadatan"]

# ==========================================================
# FUNGSI BANTU
# ==========================================================

def hitung_exog_transformed(input_t: dict, input_t1: dict) -> np.ndarray:
    """
    Transformasi input sebelum masuk model SARIMAX.

    Differencing = nilai_t - nilai_t1 (bulan sebelumnya)
    - curah_hujan : diff(1)
    - suhu        : diff(1)
    - kelembaban  : nilai asli (sudah stasioner)
    - kepadatan   : diff(1)

    Urutan output: [curah_diff, suhu_diff, kelembaban, kepadatan_diff]
    """
    return np.array([[
        input_t["curah_hujan"] - input_t1["curah_hujan"],
        input_t["suhu"]        - input_t1["suhu"],
        input_t["kelembaban"],
        input_t["kepadatan"]   - input_t1["kepadatan"],
    ]])

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

# t   = satu bulan sebelum bulan prediksi
# t-1 = dua bulan sebelum bulan prediksi
prev_2 = get_previous_months(year_pred, month_pred, n=2)
# prev_2[0] = t-1, prev_2[1] = t
year_t1, mon_t1 = prev_2[0]   # t-1
year_t,  mon_t  = prev_2[1]   # t

label_t1 = format_month_year(year_t1, mon_t1)
label_t  = format_month_year(year_t,  mon_t)

st.caption(
    f"Data yang dibutuhkan: **{label_t1}** (t-1) dan **{label_t}** (t) "
    f"→ untuk memprediksi **{month_name_pred}** (t+1)"
)

st.divider()

# ==========================================================
# INPUT DATA — 2 BULAN (t-1 dan t)
# ==========================================================

st.subheader("Masukkan Data Variabel Iklim")
st.info(
    "Masukkan nilai variabel iklim untuk 2 bulan berikut. "
    "Differencing dihitung otomatis dari selisih nilai bulan **t** "
    "terhadap bulan **t-1**. Kelembaban tidak di-differencing karena sudah stasioner."
)

periods = [
    (year_t1, mon_t1, label_t1, "t-1"),
    (year_t,  mon_t,  label_t,  "t"),
]

input_data = {}

for y, m, label, t_label in periods:
    with st.container(border=True):
        st.markdown(f"### {label} &nbsp;_({t_label})_")
        c1, c2 = st.columns(2)
        with c1:
            curah = st.number_input(
                LABEL_VARS["curah_hujan"], min_value=0.0, step=1.0,
                value=last_values["curah_hujan"],
                key=f"curah_{y}_{m}"
            )
            suhu = st.number_input(
                LABEL_VARS["suhu"], min_value=0.0, max_value=50.0, step=0.1,
                value=last_values["suhu"],
                key=f"suhu_{y}_{m}"
            )
        with c2:
            kelembaban = st.number_input(
                LABEL_VARS["kelembaban"], min_value=0.0, max_value=100.0, step=0.1,
                value=last_values["kelembaban"],
                key=f"kelembaban_{y}_{m}"
            )
            kepadatan = st.number_input(
                LABEL_VARS["kepadatan"], min_value=0.0, step=1.0,
                value=last_values["kepadatan"],
                key=f"kepadatan_{y}_{m}"
            )
        input_data[(y, m)] = {
            "curah_hujan": curah,
            "suhu":        suhu,
            "kelembaban":  kelembaban,
            "kepadatan":   kepadatan,
        }

# Tabel ringkasan
st.subheader("Ringkasan Input")
summary_rows = []
for y, m, label, t_label in periods:
    row = {"Bulan": f"{label} ({t_label})"}
    row.update({LABEL_VARS[v]: input_data[(y, m)][v] for v in EXOG_COLS})
    summary_rows.append(row)
st.dataframe(
    pd.DataFrame(summary_rows).set_index("Bulan"),
    use_container_width=True
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
        input_t1 = input_data[(year_t1, mon_t1)]   # bulan t-1
        input_t  = input_data[(year_t,  mon_t)]    # bulan t

        # 1. Hitung differencing: t - t-1
        exog_transformed = hitung_exog_transformed(input_t, input_t1)
        # shape (1, 4): [curah_diff, suhu_diff, kelembaban, kepadatan_diff]

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

        # 4. Detail transformasi
        with st.expander("🔍 Detail transformasi input (differencing)", expanded=False):
            detail_rows = []
            for var in EXOG_COLS:
                v_t1 = input_t1[var]
                v_t  = input_t[var]
                if var in DIFF_VARS:
                    nilai_model  = round(v_t - v_t1, 4)
                    transformasi = "Differencing (t − t-1)"
                else:
                    nilai_model  = round(v_t, 4)
                    transformasi = "Nilai Asli"
                detail_rows.append({
                    "Variabel":       LABEL_VARS[var],
                    f"Nilai {label_t1} (t-1)": v_t1,
                    f"Nilai {label_t} (t)":    v_t,
                    "Nilai ke Model": nilai_model,
                    "Transformasi":   transformasi,
                })
            st.dataframe(
                pd.DataFrame(detail_rows).set_index("Variabel"),
                use_container_width=True
            )
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
                label_t1, label_t,
                [input_t1["curah_hujan"], input_t["curah_hujan"]],
                [input_t1["suhu"],        input_t["suhu"]],
                [input_t1["kelembaban"],  input_t["kelembaban"]],
                [input_t1["kepadatan"],  input_t["kepadatan"]],
            ),
            use_container_width=True
        )
        st.caption(
            f"Grafik menampilkan perubahan variabel iklim dari {label_t1} (t-1) "
            f"ke {label_t} (t). Differencing dihitung dari selisih kedua nilai tersebut "
            f"untuk memprediksi kasus DBD {month_name_pred} (t+1)."
        )

        st.divider()
        st.subheader("📈 Grafik Hasil Prediksi Kasus DBD")
        
        # nilai kasus terakhir pada dataset
        last_case = float(df.iloc[-1]["kasus"])
        
        chart_df = pd.DataFrame({
            "Periode": [label_t, month_name_pred],
            "Jumlah Kasus": [last_case, pred],
            "Tipe": ["Data Aktual", "Prediksi"]
        })
        
        line = (
            alt.Chart(chart_df)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("Periode:N", title="Periode"),
                y=alt.Y("Jumlah Kasus:Q", title="Jumlah Kasus DBD"),
                color=alt.Color("Tipe:N"),
                tooltip=["Periode", "Jumlah Kasus", "Tipe"]
            )
        )

st.altair_chart(line, use_container_width=True)

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
