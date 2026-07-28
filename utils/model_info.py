from typing import Dict, Any

def get_lstm_model_info() -> Dict[str, Any]:
    """
    Mengembalikan informasi model LSTM beserta metrik evaluasi.
    Nilai metrik bisa disesuaikan dengan hasil penelitian.
    """
    info: Dict[str, Any] = {
        'Model': 'LSTM',
        'Window': 3,
        'Jumlah Variabel': 5,
        'Validasi': 'Walk Forward Validation'
    }
    return info

def get_sarimax_model_info() -> Dict[str, Any]:
    """
    Mengembalikan informasi model SARIMAX beserta metrik evaluasi.
    Nilai metrik bisa disesuaikan dengan hasil penelitian.
    """
    info: Dict[str, Any] = {
        'Order': '(1,0,0)',
        'Seasonal Order': '(1,0,0,12)',
        'Differencing Exogenous': 1,
        'Validasi': 'Walk Forward Validation',
    }
    return info
