import datetime
from datetime import timedelta
import streamlit as st
import yfinance as yf
import time

from calculations.calculations import (
    format_value
)
from visualizations.plotting import (
    plot_candlestick_daily,
)

def daily(ticker_data, selected_interval):
    # Historical data
    ph = st.empty()
    N = 60
    daily_data = yf.download(ticker_data.get_info()['symbol'],period='1d',interval=selected_interval)
    plot_daily = plot_candlestick_daily(daily_data)
    st.plotly_chart(plot_daily)
    for secs in range(N,0,-1):
        mm, ss = secs//60, secs%60
        ph.metric("Countdown", f"{mm:02d}:{ss:02d}")
        time.sleep(1)

    # # Calculate KPIs
    # last_day_data = historical_data.iloc[-1]
    # prev_day_data = historical_data.iloc[-2]
    # price_change_percent = ((last_day_data['Close'] - prev_day_data['Close']) / prev_day_data['Close']) * 100
    # volume_change = last_day_data['Volume'] - prev_day_data['Volume']
    # max_diff = last_day_data['Open'] - prev_day_data['Open']
    # min_diff = last_day_data['Close'] - prev_day_data['Close']
    # open_close = last_day_data['Open'] - prev_day_data['Close']
    #
    # col1, col2 = st.columns(2)
    # with col1:
    #     plot_candlestick_3 = plot_candlestick(historical_data, {'kendall':False,
    #                                                             'fibonacci':False,
    #                                                             'SMA200':False,
    #                                                             'SMA5':False,
    #                                                             'liquidity':False,
    #                                                             'bollinger':False})
    #     st.plotly_chart(plot_candlestick_3)
    # with col2:
    #     st.markdown(f"**Performance**: {format_value(price_change_percent)}%", unsafe_allow_html=True)
    #     st.markdown(f"**Δ Volume**: {format_value(volume_change)}", unsafe_allow_html=True)
    #     st.markdown(f"**Δ Openings**: {format_value(max_diff)}", unsafe_allow_html=True)
    #     st.markdown(f"**Δ Closing**: {format_value(min_diff)}", unsafe_allow_html=True)
    #     st.markdown(f"**Shadow Market**: {format_value(open_close)}", unsafe_allow_html=True)
