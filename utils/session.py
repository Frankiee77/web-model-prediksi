import streamlit as st
from typing import Any, Dict
from datetime import datetime

def init_session_state():
    """
    Inisialisasi state Streamlit untuk menyimpan riwayat prediksi.
    """
    if 'history' not in st.session_state:
        st.session_state['history'] = []

def add_to_history(model_name: str, pred_month: str, prediction: float):
    """
    Menambahkan catatan prediksi ke riwayat di session_state.
    """
    from datetime import datetime
    record: Dict[str, Any] = {
        'Tanggal Prediksi': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Bulan Prediksi': pred_month,
        'Model': model_name,
        'Hasil Prediksi': prediction
    }
    st.session_state['history'].append(record)
