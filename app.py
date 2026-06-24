import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ==========================================
# 1. KONFIGURASI HALAMAN WEB
# ==========================================
st.set_page_config(
    page_title="Prediksi DBD Sukabumi",
    page_icon="🦟",
    layout="wide"
)

# ==========================================
# CSS KUSTOM
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header utama */
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #e63946;
    }
    .main-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a8c4d4;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.6;
    }
    .badge {
        display: inline-block;
        background: rgba(230, 57, 70, 0.2);
        color: #ff6b6b;
        border: 1px solid rgba(230, 57, 70, 0.4);
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    /* Panel input */
    .input-section {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .month-label {
        background: #1e293b;
        color: white;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.8rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0.7rem;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.3px;
    }
    .month-label.recent {
        background: linear-gradient(135deg, #e63946, #c1121f);
    }

    /* Kartu hasil */
    .result-card {
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 0.8rem;
        border: 2px solid transparent;
        transition: transform 0.2s;
    }
    .result-card:hover { transform: translateY(-2px); }
    .result-lstm {
        background: linear-gradient(135deg, #eef2ff, #e0e7ff);
        border-color: #6366f1;
    }
    .result-sarimax {
        background: linear-gradient(135deg, #fff7ed, #ffedd5);
        border-color: #f97316;
    }
    .result-card .model-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .result-lstm .model-name { color: #4f46e5; }
    .result-sarimax .model-name { color: #ea580c; }
    .result-card .result-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.2rem;
    }
    .result-lstm .result-value { color: #3730a3; }
    .result-sarimax .result-value { color: #c2410c; }
    .result-card .result-unit {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
    }

    /* Status level */
    .status-safe { background: #dcfce7; color: #15803d; border: 1px solid #86efac; border-radius: 10px; padding: 0.8rem 1rem; margin-top: 0.5rem; font-weight: 600; font-size: 0.9rem; }
    .status-warn { background: #fef9c3; color: #a16207; border: 1px solid #fde047; border-radius: 10px; padding: 0.8rem 1rem; margin-top: 0.5rem; font-weight: 600; font-size: 0.9rem; }
    .status-danger { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; border-radius: 10px; padding: 0.8rem 1rem; margin-top: 0.5rem; font-weight: 600; font-size: 0.9rem; }

    /* Info pills */
    .info-pill {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.75rem;
        color: #475569;
        font-weight: 500;
        margin: 2px;
    }

    /* Tombol */
    .stButton > button {
        background: linear-gradient(135deg, #e63946 0%, #c1121f 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.3px;
        transition: opacity 0.2s;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Divider kustom */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, #e63946, transparent);
        border: none;
        margin: 1.5rem 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. MEMUAT MODEL & SCALER
# ==========================================
@st.cache_resource
def load_assets():
    from tensorflow.keras.models import load_model
    import statsmodels

    model_lstm = load_model('model_lstm_dbd.keras')
    model_sarimax = joblib.load('model_sarimax_dbd.pkl')
    scaler_features = joblib.load('scaler_features.pkl')
    scaler_target = joblib.load('scaler_target.pkl')
    return model_lstm, model_sarimax, scaler_features, scaler_target

try:
    model_lstm, model_sarimax, scaler_features, scaler_target = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"❌ Gagal memuat model atau scaler. Pastikan semua file berikut ada di folder yang sama dengan app.py:\n"
             f"- `model_lstm_dbd.keras`\n- `model_sarimax_dbd.pkl`\n- `.pkl`\n- `scaler_target.pkl`\n\nError: {e}")
    st.stop()


# ==========================================
# 3. FUNGSI BANTU
# ==========================================
def get_status(kasus: int) -> tuple[str, str]:
    """Return (status_html, emoji) berdasarkan jumlah kasus."""
    if kasus < 20:
        return "status-safe", "✅ Risiko Rendah — Kondisi terkendali, tetap jalankan PSN rutin."
    elif kasus < 50:
        return "status-warn", "⚠️ Risiko Sedang — Tingkatkan pemantauan dan kegiatan PSN 3M Plus."
    else:
        return "status-danger", "🚨 Risiko Tinggi — Potensi KLB! Aktifkan respons darurat segera."


def make_chart(bulan_historis: list[str], kasus_historis: list[float],
               bulan_prediksi: str, pred_lstm: int, pred_sarimax: int) -> plt.Figure:
    """Buat grafik perbandingan prediksi LSTM vs SARIMAX."""

    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor('#f8fafc')

    ax = fig.add_subplot(111)
    ax.set_facecolor('#f8fafc')

    # --- Data historis ---
    x_hist = list(range(len(bulan_historis)))
    ax.plot(x_hist, kasus_historis, color='#475569', linewidth=2.2,
            marker='o', markersize=6, markerfacecolor='white',
            markeredgewidth=2, label='Data Historis', zorder=3)

    # Isi area di bawah kurva historis
    ax.fill_between(x_hist, kasus_historis, alpha=0.08, color='#475569')

    # --- Titik prediksi ---
    x_pred = len(bulan_historis)

    # Garis putus-putus dari titik terakhir historis ke prediksi
    x_last = x_hist[-1]
    y_last = kasus_historis[-1]

    ax.plot([x_last, x_pred], [y_last, pred_lstm],
            color='#6366f1', linewidth=1.8, linestyle='--', alpha=0.7, zorder=2)
    ax.plot([x_last, x_pred], [y_last, pred_sarimax],
            color='#f97316', linewidth=1.8, linestyle='--', alpha=0.7, zorder=2)

    # Marker prediksi LSTM
    ax.scatter(x_pred, pred_lstm, color='#6366f1', s=140, zorder=5,
               edgecolors='white', linewidths=2)
    ax.annotate(f'LSTM\n{pred_lstm} kasus',
                xy=(x_pred, pred_lstm),
                xytext=(x_pred + 0.15, pred_lstm + max(kasus_historis) * 0.06),
                fontsize=9, fontweight='600', color='#4f46e5',
                fontfamily='sans-serif',
                arrowprops=dict(arrowstyle='->', color='#6366f1', lw=1.2))

    # Marker prediksi SARIMAX
    ax.scatter(x_pred, pred_sarimax, color='#f97316', s=140, zorder=5,
               edgecolors='white', linewidths=2)
    offset_dir = -1 if pred_sarimax >= pred_lstm else 1
    ax.annotate(f'SARIMAX\n{pred_sarimax} kasus',
                xy=(x_pred, pred_sarimax),
                xytext=(x_pred + 0.15, pred_sarimax + offset_dir * max(kasus_historis) * 0.06),
                fontsize=9, fontweight='600', color='#c2410c',
                fontfamily='sans-serif',
                arrowprops=dict(arrowstyle='->', color='#f97316', lw=1.2))

    # --- Garis ambang batas ---
    ax.axhline(y=50, color='#ef4444', linewidth=1, linestyle=':', alpha=0.6)
    ax.text(len(bulan_historis) + 0.5, 51, 'Ambang KLB (50)', fontsize=7.5,
            color='#ef4444', alpha=0.8)
    ax.axhline(y=20, color='#eab308', linewidth=1, linestyle=':', alpha=0.5)
    ax.text(len(bulan_historis) + 0.5, 21, 'Risiko Sedang (20)', fontsize=7.5,
            color='#c4880f', alpha=0.8)

    # --- Area prediksi ---
    ax.axvspan(x_pred - 0.5, x_pred + 0.7, alpha=0.06, color='#6366f1',
               label='_nolegend_')
    ax.text(x_pred, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 5,
            'Prediksi', fontsize=8, color='#6366f1', ha='center', alpha=0.7, style='italic')

    # --- Sumbu & label ---
    all_x_labels = bulan_historis + [bulan_prediksi]
    all_x = list(range(len(all_x_labels)))
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_x_labels, fontsize=9, color='#475569')
    ax.set_ylabel('Jumlah Kasus DBD', fontsize=10, color='#475569', labelpad=10)
    ax.set_xlabel('Bulan', fontsize=10, color='#475569', labelpad=8)

    ax.tick_params(colors='#94a3b8', length=4)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')
    ax.yaxis.grid(True, color='#e2e8f0', linewidth=0.8, linestyle='--')
    ax.set_axisbelow(True)

    # --- Legenda ---
    legend_elements = [
        mpatches.Patch(facecolor='#475569', alpha=0.6, label='Data Historis'),
        mpatches.Patch(facecolor='#6366f1', label=f'Prediksi LSTM ({pred_lstm} kasus)'),
        mpatches.Patch(facecolor='#f97316', label=f'Prediksi SARIMAX ({pred_sarimax} kasus)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8.5,
              framealpha=0.8, edgecolor='#e2e8f0', fancybox=True)

    ax.set_title('Perbandingan Prediksi Kasus DBD: LSTM vs SARIMAX',
                 fontsize=13, fontweight='700', color='#1e293b', pad=15)

    plt.tight_layout()
    return fig


# ==========================================
# 4. HEADER APLIKASI
# ==========================================
st.markdown("""
<div class="main-header">
    <div class="badge">🦟 Sistem Kewaspadaan DBD</div>
    <h1>Prediksi DBD Kota Sukabumi</h1>
    <p>
        Perbandingan dua model prediksi — <strong>LSTM</strong> (Deep Learning) dan <strong>SARIMAX</strong> (Statistical) — 
        menggunakan 4 variabel: curah hujan, suhu, kelembaban, dan kepadatan penduduk pada 3 bulan terakhir.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 5. LAYOUT DUA KOLOM: INPUT & HASIL
# ==========================================
col_input, col_result = st.columns([1.4, 1], gap="large")

with col_input:
    st.markdown('<div class="section-title">📋 Input Data Historis (3 Bulan Terakhir)</div>',
                unsafe_allow_html=True)

    BULAN_OPTIONS = [
        "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember"
    ]
    TAHUN_OPTIONS = list(range(2022, 2027))

    # ---- Bulan t-3 ----
    with st.expander("📅 Bulan t-3 (Paling Lama)", expanded=True):
        c1a, c1b = st.columns(2)
        with c1a:
            bulan_t3 = st.selectbox("Bulan", BULAN_OPTIONS, index=0, key="bln_t3")
        with c1b:
            tahun_t3 = st.selectbox("Tahun", TAHUN_OPTIONS, index=2, key="thn_t3")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kasus_1 = st.number_input("Kasus DBD", min_value=0, key="k1",
                                      help="Jumlah kasus DBD terkonfirmasi")
        with c2:
            hujan_1 = st.number_input("Curah Hujan (mm)", min_value=0.0, key="h1")
        with c3:
            suhu_1 = st.number_input("Suhu (°C)", min_value=15.0, max_value=45.0,
                                     step=0.1, key="s1")
        with c4:
            lembab_1 = st.number_input("Kelembaban (%)", min_value=0.0, max_value=100.0,
                                       key="l1")

        padat_1 = st.number_input("Kepadatan Penduduk (jiwa/km²)",
                                  min_value=0, key="p1")

    # ---- Bulan t-2 ----
    with st.expander("📅 Bulan t-2", expanded=True):
        c2a, c2b = st.columns(2)
        with c2a:
            bulan_t2 = st.selectbox("Bulan", BULAN_OPTIONS, index=1, key="bln_t2")
        with c2b:
            tahun_t2 = st.selectbox("Tahun", TAHUN_OPTIONS, index=2, key="thn_t2")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kasus_2 = st.number_input("Kasus DBD", min_value=0, key="k2")
        with c2:
            hujan_2 = st.number_input("Curah Hujan (mm)", min_value=0.0, key="h2")
        with c3:
            suhu_2 = st.number_input("Suhu (°C)", min_value=15.0, max_value=45.0,
                                    step=0.1, key="s2")
        with c4:
            lembab_2 = st.number_input("Kelembaban (%)", min_value=0.0, max_value=100.0,
                                       key="l2")

        padat_2 = st.number_input("Kepadatan Penduduk (jiwa/km²)",
                                  min_value=0, key="p2")

    # ---- Bulan t-1 ----
    with st.expander("📅 Bulan t-1 (Paling Baru)", expanded=True):
        c3a, c3b = st.columns(2)
        with c3a:
            bulan_t1 = st.selectbox("Bulan", BULAN_OPTIONS, index=2, key="bln_t1")
        with c3b:
            tahun_t1 = st.selectbox("Tahun", TAHUN_OPTIONS, index=2, key="thn_t1")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kasus_3 = st.number_input("Kasus DBD", min_value=0, key="k3")
        with c2:
            hujan_3 = st.number_input("Curah Hujan (mm)", min_value=0.0, key="h3")
        with c3:
            suhu_3 = st.number_input("Suhu (°C)", min_value=15.0, max_value=45.0, step=0.1, key="s3")
        with c4:
            lembab_3 = st.number_input("Kelembaban (%)", min_value=0.0, max_value=100.0, key="l3")

        padat_3 = st.number_input("Kepadatan Penduduk (jiwa/km²)",
                                  min_value=0, key="p3")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    predict_btn = st.button("🚀 Jalankan Prediksi LSTM & SARIMAX", use_container_width=True)


# ==========================================
# KOLOM KANAN: HASIL
# ==========================================
with col_result:
    st.markdown('<div class="section-title">📊 Hasil Prediksi Bulan Berikutnya</div>',
                unsafe_allow_html=True)

    if not predict_btn:
        st.info("← Isi data historis di sebelah kiri, lalu klik tombol **Prediksi** untuk melihat hasil perbandingan model.")
        st.markdown("""
        **Variabel yang digunakan:**
        <span class="info-pill">🌧️ Curah Hujan</span>
        <span class="info-pill">🌡️ Suhu Udara</span>
        <span class="info-pill">💧 Kelembaban</span>
        <span class="info-pill">👥 Kepadatan</span>

        **Model yang dibandingkan:**
        <span class="info-pill">🧠 LSTM (Deep Learning)</span>
        <span class="info-pill">📈 SARIMAX (Statistik)</span>
        """, unsafe_allow_html=True)

    else:
        # ==========================================
        # 6. PROSES PREDIKSI
        # ==========================================

        # --- Ambil batas range training dari  ---
        # [curah_hujan, suhu, kelembaban, kepadatan]
        # data_min_: [0, 22.31, 77.07, 6995]
        # data_max_: [2740, 25.45, 91.44, 7820]
        feat_min = scaler_features.data_min_   # shape (4,)
        feat_max = scaler_features.data_max_   # shape (4,)

        # Clip input ke range training agar hasil scale tetap 0–1
        # (input di luar range menyebabkan ekstrapolasi → prediksi meledak)
        def clip_features(h, s, l, p):
            arr = np.array([h, s, l, p])
            return np.clip(arr, feat_min, feat_max)

        raw_t3 = clip_features(hujan_1, suhu_1, lembab_1, padat_1)
        raw_t2 = clip_features(hujan_2, suhu_2, lembab_2, padat_2)
        raw_t1 = clip_features(hujan_3, suhu_3, lembab_3, padat_3)

        # Peringatan jika ada nilai di luar range training
        input_raw = np.array([
            [hujan_1, suhu_1, lembab_1, padat_1],
            [hujan_2, suhu_2, lembab_2, padat_2],
            [hujan_3, suhu_3, lembab_3, padat_3]
        ])
        out_of_range = np.any(input_raw < feat_min, axis=0) | np.any(input_raw > feat_max, axis=0)
        col_names = ['Curah Hujan', 'Suhu', 'Kelembaban', 'Kepadatan']
        oor_cols = [col_names[i] for i in range(4) if out_of_range[i]]
        if oor_cols:
            range_info = ', '.join([f'{col_names[i]} ({feat_min[i]:.1f}–{feat_max[i]:.1f})' 
                                    for i in range(4) if out_of_range[i]])
            st.warning(f"⚠️ **Input di luar rentang data training** untuk: **{', '.join(oor_cols)}**. "
                       f"Nilai akan di-clip ke rentang training: {range_info}. "
                       f"Hasil prediksi tetap valid namun kurang akurat untuk kondisi ekstrem.")

        # --- Scale variabel input (4 fitur) ---
        input_data_4f = np.array([raw_t3, raw_t2, raw_t1])  # (3, 4) sudah di-clip
        input_scaled_4f = scaler_features.transform(input_data_4f)  # (3, 4), range 0–1

        # --- Scale kasus historis via scaler_target (untuk lag LSTM) ---
        kasus_arr = np.array([[kasus_1], [kasus_2], [kasus_3]])
        kasus_clipped = np.clip(kasus_arr, scaler_target.data_min_, scaler_target.data_max_)
        kasus_scaled = scaler_target.transform(kasus_clipped)  # (3, 1)

        # --- Gabungkan untuk LSTM: [Kasus | Hujan, Suhu, Lembab, Padat] → (3, 5) ---
        input_scaled = np.hstack([kasus_scaled, input_scaled_4f])

        # ---- Prediksi LSTM ----
        input_lstm = input_scaled.reshape((1, 3, 5))  # (1 sampel, 3 time-steps, 5 fitur)
        pred_lstm_scaled = model_lstm.predict(input_lstm, verbose=0)
        pred_lstm_actual = scaler_target.inverse_transform(pred_lstm_scaled)
        hasil_lstm = max(0, int(np.round(pred_lstm_actual[0][0])))

        # ---- Prediksi SARIMAX ----
        # SARIMAX dilatih dengan exog yang SUDAH di-scale (0–1), bukan nilai mentah
        # Gunakan input_scaled_4f baris terakhir (t-1) sebagai exog untuk forecast t+1
        exog_next_scaled = input_scaled_4f[[-1], :]  # shape (1, 4) — sudah scaled 0–1
        try:
            pred_sarimax_result = model_sarimax.forecast(steps=1, exog=exog_next_scaled)
            hasil_sarimax = max(0, int(np.round(float(pred_sarimax_result.iloc[0]))))
        except Exception as e:
            try:
                fc = model_sarimax.get_forecast(steps=1, exog=exog_next_scaled)
                hasil_sarimax = max(0, int(np.round(float(fc.predicted_mean.iloc[0]))))
            except Exception as e2:
                st.warning(f"SARIMAX forecasting error: {e2}")
                hasil_sarimax = max(0, int(np.round(float(model_sarimax.fittedvalues.iloc[-1]))))

        # ==========================================
        # 7. TAMPILAN HASIL
        # ==========================================
        st.markdown(f"""
        <div class="result-card result-lstm">
            <div class="model-name">🧠 LSTM — Deep Learning</div>
            <div class="result-value">{hasil_lstm}</div>
            <div class="result-unit">kasus prediksi</div>
        </div>
        """, unsafe_allow_html=True)

        status_cls_lstm, status_txt_lstm = get_status(hasil_lstm)
        st.markdown(f'<div class="{status_cls_lstm}">{status_txt_lstm}</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card result-sarimax">
            <div class="model-name">📈 SARIMAX — Statistik</div>
            <div class="result-value">{hasil_sarimax}</div>
            <div class="result-unit">kasus prediksi</div>
        </div>
        """, unsafe_allow_html=True)

        status_cls_sx, status_txt_sx = get_status(hasil_sarimax)
        st.markdown(f'<div class="{status_cls_sx}">{status_txt_sx}</div>',
                    unsafe_allow_html=True)

        # Selisih prediksi
        selisih = abs(hasil_lstm - hasil_sarimax)
        rata2 = (hasil_lstm + hasil_sarimax) / 2
        st.markdown(f"""
        <br>
        <div style="background:#f1f5f9; border-radius:10px; padding:0.8rem 1rem; font-size:0.85rem; color:#475569;">
            <strong>Selisih antar model:</strong> {selisih} kasus &nbsp;|&nbsp;
            <strong>Rata-rata:</strong> {rata2:.0f} kasus
        </div>
        """, unsafe_allow_html=True)

        # ==========================================
        # 8. GRAFIK PERBANDINGAN (FULL WIDTH)
        # ==========================================
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.subheader("📉 Grafik Perbandingan Prediksi")

        label_t3 = f"{bulan_t3[:3]} {tahun_t3}"
        label_t2 = f"{bulan_t2[:3]} {tahun_t2}"
        label_t1 = f"{bulan_t1[:3]} {tahun_t1}"

        # Estimasi bulan prediksi (bulan setelah t-1)
        idx_next = (BULAN_OPTIONS.index(bulan_t1) + 1) % 12
        tahun_next = tahun_t1 + 1 if idx_next == 0 else tahun_t1
        label_pred = f"{BULAN_OPTIONS[idx_next][:3]} {tahun_next}"

        fig = make_chart(
            bulan_historis=[label_t3, label_t2, label_t1],
            kasus_historis=[float(kasus_1), float(kasus_2), float(kasus_3)],
            bulan_prediksi=label_pred,
            pred_lstm=hasil_lstm,
            pred_sarimax=hasil_sarimax
        )
        st.pyplot(fig, use_container_width=True)

        # ==========================================
        # 9. TABEL RINGKASAN
        # ==========================================
        st.subheader("📋 Ringkasan Perbandingan Model")
        df_summary = pd.DataFrame({
            "Model": ["LSTM (Deep Learning)", "SARIMAX (Statistik)"],
            "Prediksi Kasus": [hasil_lstm, hasil_sarimax],
            "Tingkat Risiko": [
                "🟢 Rendah" if hasil_lstm < 20 else ("🟡 Sedang" if hasil_lstm < 50 else "🔴 Tinggi"),
                "🟢 Rendah" if hasil_sarimax < 20 else ("🟡 Sedang" if hasil_sarimax < 50 else "🔴 Tinggi"),
            ],
            "Pendekatan": ["Sequence Pattern (LSTM)", "Time Series + Exog (ARIMA)"],
        })
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        st.caption("⚠️ Hasil prediksi bersifat indikatif untuk mendukung keputusan dinas kesehatan. "
                   "Selalu verifikasi dengan data lapangan terkini.")
