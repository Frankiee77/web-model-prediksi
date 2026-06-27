import streamlit as st

st.set_page_config(
    page_title="Prediksi DBD Kota Sukabumi",
    page_icon="🦟",
    layout="wide"
)

st.title("Prediksi Kasus DBD Kota Sukabumi")

st.markdown("""
Aplikasi ini dikembangkan untuk memprediksi jumlah kasus
Demam Berdarah Dengue (DBD) di Kota Sukabumi menggunakan dua model:

- 🧠 Long Short-Term Memory (LSTM)
- 📈 SARIMAX

Silakan pilih model pada sidebar.
""")
