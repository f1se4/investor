import datetime
from datetime import timedelta
import streamlit as st

from calculations.calculations import (
    format_value
)
from visualizations.plotting import (
    plot_candlestick,
)

def daily(ticker_data):
    # Historical data
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    five_days_ago = (datetime.datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')
    historical_data = ticker_data.history(start=five_days_ago, end=today)

    # Calculate KPIs
    last_day_data = historical_data.iloc[-1]
    prev_day_data = historical_data.iloc[-2]
    price_change_percent = ((last_day_data['Close'] - prev_day_data['Close']) / prev_day_data['Close']) * 100
    volume_change = last_day_data['Volume'] - prev_day_data['Volume']
    max_diff = last_day_data['Open'] - prev_day_data['Open']
    min_diff = last_day_data['Close'] - prev_day_data['Close']
    open_close = last_day_data['Open'] - prev_day_data['Close']

    col1, col2 = st.columns(2)
    with col1:
        plot_candlestick_3 = plot_candlestick(historical_data, {'kendall':False,
                                                                'fibonacci':False,
                                                                'liquidity':False,
                                                                'bollinger':False})
        st.plotly_chart(plot_candlestick_3)
    with col2:
        st.markdown(f"**Performance**: {format_value(price_change_percent)}%", unsafe_allow_html=True)
        st.markdown(f"**Δ Volume**: {format_value(volume_change)}", unsafe_allow_html=True)
        st.markdown(f"**Δ Openings**: {format_value(max_diff)}", unsafe_allow_html=True)
        st.markdown(f"**Δ Closing**: {format_value(min_diff)}", unsafe_allow_html=True)
        st.markdown(f"**Shadow Market**: {format_value(open_close)}", unsafe_allow_html=True)
