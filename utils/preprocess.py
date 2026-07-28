from typing import List

def compute_diff_exog(prev_value: float, curr_value: float) -> float:
    """
    Menghitung selisih orde-1 antara nilai periode sekarang dan sebelumnya.
    """
    return curr_value - prev_value

def prepare_lstm_input(values: List[float]) -> List[float]:
    """
    Tempat untuk melakukan pra-pemrosesan data masukan LSTM jika diperlukan.
    Misalnya, mengubah format list ke numpy array, menskalakan, dsb.
    (Placeholder, implementasi sesuai kebutuhan).
    """
    # Di sini seharusnya penerapan scaler atau transformasi lain jika diperlukan.
    return values
