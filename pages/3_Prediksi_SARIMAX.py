import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from datetime import datetime
from scipy.special import inv_boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
    "kasus":       float(last_row["kasus"]),
    "curah_hujan": float(last_row["curah_hujan"]),
    "suhu":        float(last_row["suhu"]),
    "kelembaban":  float(last_row["kelembaban"]),
    "kepadatan":   float(last_row["kepadatan"]),
}

# Parameter model — harus identik dengan saat training
ORDER          = artifacts.get("order",          (1, 0, 0))
SEASONAL_ORDER = artifacts.get("seasonal_order", (1, 0, 0, 12))

init_session_state()

# ==========================================================
# KONSTANTA
# ==========================================================

EXOG_COLS = ["curah_hujan", "suhu", "kelembaban", "kepadatan"]

LABEL_VARS = {
    "kasus":       "Kasus DBD",
    "curah_hujan": "Curah Hujan (mm)",
    "suhu":        "Suhu (°C)",
    "kelembaban":  "Kelembaban (%)",
    "kepadatan":   "Kepadatan Penduduk (jiwa/km²)",
}

# Variabel eksogen yang di-differencing
DIFF_VARS = ["curah_hujan", "suhu", "kepadatan"]

# ==========================================================
# FUNGSI BANTU
# ==========================================================

def hitung_exog_transformed(input_t: dict, input_t1: dict) -> np.ndarray:
    """
    Transformasi eksogen sebelum masuk model SARIMAX.
    Differencing = nilai_t - nilai_t1
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
# INPUT DATA — 2 BULAN (t-1 dan t), SEMUA VARIABEL
# ==========================================================

st.subheader("Masukkan Data Historis")
st.info(
    "Masukkan seluruh nilai variabel untuk 2 bulan berikut. "
    "Kasus DBD dan kelembaban dimasukkan sebagai nilai asli. "
    "Curah hujan, suhu, dan kepadatan akan di-differencing "
    "secara otomatis (nilai t − nilai t-1)."
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
            kasus = st.number_input(
                LABEL_VARS["kasus"], min_value=0, step=1,
                value=int(last_values["kasus"]),
                key=f"kasus_{y}_{m}"
            )
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
            "kasus":       float(kasus),
            "curah_hujan": curah,
            "suhu":        suhu,
            "kelembaban":  kelembaban,
            "kepadatan":   kepadatan,
        }

# Tabel ringkasan
st.subheader("Ringkasan Input")
all_vars = ["kasus"] + EXOG_COLS
summary_rows = []
for y, m, label, t_label in periods:
    row = {"Bulan": f"{label} ({t_label})"}
    row.update({LABEL_VARS[v]: input_data[(y, m)][v] for v in all_vars})
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
        input_t1 = input_data[(year_t1, mon_t1)]
        input_t  = input_data[(year_t,  mon_t)]

        # 1. Transformasi Box-Cox pada kasus (endog)
        #    Kasus bulan t sebagai titik awal endog baru
        from scipy.stats import boxcox as scipy_boxcox
        kasus_t_trans = float(
            inv_boxcox(
                # Gunakan lambda yang sama dengan training
                # Box-Cox manual: ((x^lambda) - 1) / lambda
                (input_t["kasus"] ** fitted_lambda - 1) / fitted_lambda
                if fitted_lambda != 0
                else np.log(input_t["kasus"] + 1e-6),
                fitted_lambda
            )
        )
        # Cara yang benar: transformasi langsung dengan fitted_lambda
        if fitted_lambda != 0:
            kasus_t_trans = (input_t["kasus"] ** fitted_lambda - 1) / fitted_lambda
        else:
            kasus_t_trans = np.log(max(input_t["kasus"], 1e-6))

        # 2. Hitung differencing eksogen: t - t-1
        exog_transformed = hitung_exog_transformed(input_t, input_t1)
        # shape (1, 4): [curah_diff, suhu_diff, kelembaban, kepadatan_diff]

        # 3. Append data baru ke model lalu forecast 1 langkah
        #    Ini memperbarui state model dengan data bulan t
        #    sebelum memprediksi t+1
        result_updated = result.append(
            endog=[kasus_t_trans],
            exog=exog_transformed,
            refit=False
        )

        # 4. Siapkan eksogen untuk langkah prediksi (t+1)
        #    Untuk t+1 kita tidak punya data aktual eksogen,
        #    sehingga gunakan nilai t sebagai aproksimasi
        #    (pengguna bisa mengubah ini jika ada estimasi)
        exog_next = exog_transformed.copy()

        # 5. Forecast t+1 dalam skala Box-Cox
        pred_transformed = result_updated.forecast(
            steps=1, exog=exog_next
        )[0]

        # 6. Inverse Box-Cox ke skala kasus asli
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

        # 7. Detail transformasi
        with st.expander("🔍 Detail transformasi input", expanded=False):
            detail_rows = []
            # Kasus
            detail_rows.append({
                "Variabel":              LABEL_VARS["kasus"],
                f"Nilai {label_t1} (t-1)": input_t1["kasus"],
                f"Nilai {label_t} (t)":    input_t["kasus"],
                "Nilai ke Model":        round(kasus_t_trans, 4),
                "Transformasi":          f"Box-Cox (λ={fitted_lambda:.4f})",
            })
            # Eksogen
            for i, var in enumerate(EXOG_COLS):
                v_t1 = input_t1[var]
                v_t  = input_t[var]
                if var in DIFF_VARS:
                    nilai_model  = round(v_t - v_t1, 4)
                    transformasi = "Differencing (t − t-1)"
                else:
                    nilai_model  = round(v_t, 4)
                    transformasi = "Nilai Asli"
                detail_rows.append({
                    "Variabel":              LABEL_VARS[var],
                    f"Nilai {label_t1} (t-1)": v_t1,
                    f"Nilai {label_t} (t)":    v_t,
                    "Nilai ke Model":        nilai_model,
                    "Transformasi":          transformasi,
                })
            st.dataframe(
                pd.DataFrame(detail_rows).set_index("Variabel"),
                use_container_width=True
            )
            st.caption(
                f"Prediksi dalam skala Box-Cox (λ={fitted_lambda:.4f}) "
                f"= {pred_transformed:.4f} → skala kasus asli = {pred}."
            )

        # 8. Tambah ke riwayat
        add_to_history("SARIMAX", month_name_pred, pred)

        st.divider()
        st.subheader("📈 Grafik Hasil Prediksi Kasus DBD")
        
        chart_df = pd.DataFrame({
            "Periode": [
                label_t1,
                label_t,
                month_name_pred
            ],
            "Jumlah Kasus": [
                input_t1["kasus"],
                input_t["kasus"],
                pred
            ],
            "Tipe": [
                "Data Aktual",
                "Data Aktual",
                "Prediksi"
            ]
        })
        
        line = (
            alt.Chart(chart_df)
            .mark_line(
                point=True,
                strokeWidth=3,
                strokeDash=[6, 4]
            )
            .encode(
                x=alt.X("Periode:N", title="Periode"),
                y=alt.Y("Jumlah Kasus:Q", title="Jumlah Kasus DBD"),
                color=alt.Color(
                    "Tipe:N",
                    scale=alt.Scale(
                        domain=["Data Aktual", "Prediksi"],
                        range=["#1f77b4", "#d62728"]
                    )
                ),
                tooltip=[
                    alt.Tooltip("Periode:N"),
                    alt.Tooltip("Jumlah Kasus:Q", title="Jumlah Kasus"),
                    alt.Tooltip("Tipe:N")
                ]
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
