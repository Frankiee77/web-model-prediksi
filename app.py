import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==========================================
# 1. KONFIGURASI HALAMAN WEB
# ==========================================
st.set_page_config(
    page_title="Prediksi DBD Sukabumi",
    page_icon="🦟",
    layout="centered"
)

# ==========================================
# 2. MEMUAT MODEL & SCALER (CACHE)
# ==========================================
# st.cache_resource digunakan agar model tidak dimuat ulang setiap kali pengguna mengetik angka
@st.cache_resource
def load_assets():
    # Ganti nama file sesuai dengan file yang Anda unduh dari Colab
    model = load_model('model_lstm.keras')
    scaler_all = joblib.load('scaler_all.pkl')
    scaler_target = joblib.load('scaler_target.pkl')
    return model, scaler_all, scaler_target

try:
    model, scaler_all, scaler_target = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Gagal memuat model atau scaler. Pastikan file .keras dan .pkl ada di folder yang sama. Error: {e}")

# ==========================================
# 3. TAMPILAN ANTARMUKA (HEADER)
# ==========================================
st.title("🦟 Sistem Prediksi DBD Kota Sukabumi")
st.markdown("""
Aplikasi berbasis *Deep Learning* (LSTM) untuk memprediksi jumlah kasus Demam Berdarah Dengue (DBD) bulan depan. 
Silakan masukkan data curah hujan, kepadatan penduduk, dan riwayat kasus pada **3 bulan terakhir**.
""")
st.divider()

if assets_loaded:
    # ==========================================
    # 4. FORM INPUT DATA (3 KOLOM UNTUK LAG 3 BULAN)
    # ==========================================
    st.subheader("Input Data Historis (3 Bulan Terakhir)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Bulan (t-3)**")
        kasus_1 = st.number_input("Kasus", min_value=0, value=0, key="k1")
        hujan_1 = st.number_input("Curah Hujan (mm)", min_value=0.0, value=0.0, key="h1")
        padat_1 = st.number_input("Kepadatan", min_value=0, value=7500, key="p1")

    with col2:
        st.markdown("**Bulan (t-2)**")
        kasus_2 = st.number_input("Kasus", min_value=0, value=0, key="k2")
        hujan_2 = st.number_input("Curah Hujan (mm)", min_value=0.0, value=0.0, key="h2")
        padat_2 = st.number_input("Kepadatan", min_value=0, value=7500, key="p2")

    with col3:
        st.markdown("**Bulan (t-1) / Terakhir**")
        kasus_3 = st.number_input("Kasus", min_value=0, value=0, key="k3")
        hujan_3 = st.number_input("Curah Hujan (mm)", min_value=0.0, value=0.0, key="h3")
        padat_3 = st.number_input("Kepadatan", min_value=0, value=7500, key="p3")

    st.divider()

    # ==========================================
    # 5. PROSES PREDIKSI
    # ==========================================
    if st.button("🚀 Prediksi Kasus Bulan Depan", use_container_width=True):
        
        # Menyusun data sesuai urutan fitur saat training: [Kasus, Curah Hujan, Kepadatan]
        input_data = np.array([
            [kasus_1, hujan_1, padat_1],
            [kasus_2, hujan_2, padat_2],
            [kasus_3, hujan_3, padat_3]
        ])
        
        # Menerjemahkan angka riil ke skala 0-1 menggunakan scaler_all
        input_scaled = scaler_all.transform(input_data)
        
        # Membentuk ulang dimensi array menjadi 3D untuk LSTM: (1 sampel, 3 time-steps, 3 fitur)
        input_lstm = input_scaled.reshape((1, 3, 3))
        
        # Eksekusi prediksi
        prediksi_scaled = model.predict(input_lstm)
        
        # Mengembalikan skala hasil prediksi menjadi angka kasus nyata
        prediksi_aktual = scaler_target.inverse_transform(prediksi_scaled)
        
        # Membulatkan hasil (karena jumlah orang tidak mungkin desimal)
        hasil_final = int(np.round(prediksi_aktual[0][0]))
        
        # Mencegah hasil prediksi minus
        if hasil_final < 0:
            hasil_final = 0

        # Menampilkan hasil
        st.success("Berhasil melakukan kalkulasi!")
        st.metric(label="Prediksi Jumlah Kasus Bulan Depan", value=f"{hasil_final} Orang")
        
        # Menambahkan peringatan jika kasus diprediksi tinggi
        if hasil_final >= 50:
            st.warning("⚠️ **Waspada Wabah:** Prediksi menunjukkan lonjakan kasus. Segera tingkatkan upaya Pemberantasan Sarang Nyamuk (PSN) 3M Plus.")
