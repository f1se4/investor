import streamlit as st
import yfinance as yf
from pytz import timezone, utc

from visualizations.plotting import (
    plot_candlestick_daily,
)

def daily(ticker_data, selected_interval):
    # Historical data
    utc_tz = utc
    cest_tz = timezone('Europe/Paris')  # Assuming CEST is Europe/Paris time zone
    daily_data = yf.download(ticker_data.get_info()['symbol'],period='1d',interval=selected_interval)
    plot_daily = plot_candlestick_daily(daily_data)
    st.plotly_chart(plot_daily)
