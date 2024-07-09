import streamlit as st
import yfinance as yf

from visualizations.plotting import (
    plot_candlestick_daily,
)

def daily(ticker_data, selected_interval):
    # Historical data
    daily_data = yf.download(ticker_data.get_info()['symbol'],period='1d',interval=selected_interval)

    # # Convert index (date) to CEST
    try:
        daily_data.index = daily_data.index.tz_convert('CET')
    except:
        pass

    plot_daily = plot_candlestick_daily(daily_data)
    st.plotly_chart(plot_daily)
