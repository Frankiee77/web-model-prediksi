import pandas as pd
import altair as alt
import streamlit as st

def plot_lstm_prediction(months: list, cases: list, prediction: float):
    """
    Membuat grafik line chart untuk data input LSTM (kasus historis) 
    dan nilai prediksi.
    """
    all_months = months + ["Prediksi"]
    all_values = cases + [prediction]
    df = pd.DataFrame({'Bulan': all_months, 'Kasus': all_values})
    # Gunakan ordinal axis agar urutan bulan tetap seperti input
    chart = alt.Chart(df).mark_line(point=True, color='blue').encode(
        x=alt.X('Bulan:N', sort=None, title='Bulan'),
        y=alt.Y('Kasus:Q', title='Jumlah Kasus')
    ).properties(
        title='Prediksi Kasus DBD',
        width=600,
        height=400
    )
    return chart

def plot_sarimax_inputs(prev_month: str, curr_month: str, 
                        curah: list, suhu: list, kelembaban: list, kepadatan: list):
    """
    Membuat grafik line chart untuk input variabel eksogen SARIMAX 
    (dua periode: t-1 dan t).
    """
    # DataFrame dengan dua periode (prev_month, curr_month)
    df = pd.DataFrame({
        'Bulan': [prev_month, curr_month],
        'Curah Hujan': curah,
        'Suhu': suhu,
        'Kelembaban': kelembaban,
        'Kepadatan': kepadatan
    })
    # Lipat data untuk banyak garis
    df_melted = df.melt(id_vars=['Bulan'], 
                        value_vars=['Curah Hujan', 'Suhu', 'Kelembaban', 'Kepadatan'],
                        var_name='Variabel', value_name='Nilai')
    chart = alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X('Bulan:N', sort=None, title='Bulan'),
        y=alt.Y('Nilai:Q', title='Nilai'),
        color=alt.Color('Variabel:N', title='Variabel')
    ).properties(
        title='Input Variabel Eksogen',
        width=600,
        height=400
    )
    return chart
