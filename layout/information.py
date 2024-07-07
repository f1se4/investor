import streamlit as st
import yfinance as yf
from visualizations.plotting import (
    plot_rendimiento,
    plot_per_gauge
)
from calculations.calculations import mostrar_informacion_general, obtener_constituyentes_y_pesos

def information(stock, category):
    # Show performance plot
    st.subheader('Performance')
    rendimiento_plot = plot_rendimiento(stock)
    st.plotly_chart(rendimiento_plot, use_container_width=True)
    # Ticker data from yfinance
    ticker_data = yf.Ticker(stock)

    try:
        PER = ticker_data.info['trailingPE']
        st.plotly_chart(plot_per_gauge(PER))
    except:
        pass

    # Show general information
    st.subheader('General Information')
    mostrar_informacion_general(ticker_data)

    if category == 'Indexados':
        st.subheader('Constituents')
        st.dataframe(obtener_constituyentes_y_pesos(stock))

    return ticker_data
