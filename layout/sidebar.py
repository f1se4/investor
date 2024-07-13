import datetime
from datetime import timedelta
import streamlit as st
from PIL import Image
import json

from calculations.calculations import get_company_name

# Function to read tickers from JSON
def read_tickers():
    with open('tickers.json', 'r') as file:
        tickers = json.load(file)
    return tickers

# Function to configure sidebar
def configure_sidebar():
    st.sidebar.markdown("""
    <style>
    .link-text {
        color: #b0bec5; /* Color metalizado */
        font-weight: bold;
        text-decoration: none;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    logo = Image.open('assets/logo.jpeg')
    st.sidebar.image(logo)
    
    st.sidebar.header("Select Ticker")

    # Read tickers from file
    tickers_data = read_tickers()

    refresh_data = st.sidebar.checkbox('Refresh Data (60s)', value=True)

    # Select category
    category = st.sidebar.selectbox("Category", list(tickers_data.keys()))
    
    # Select subcategory
    subcategory = st.sidebar.selectbox("Subcategory", list(tickers_data[category].keys()))
    
    # Select ticker
    tickers = []
    tickers_list = tickers_data[category][subcategory]
    for ticker in tickers_list:
        ticker = ticker + " - " + str(get_company_name(ticker))
        tickers.append(ticker)
    selected_ticker = st.sidebar.selectbox("Selecciona un Ticker", tickers)
    stock = selected_ticker.split(' - ')[0]

    st.sidebar.header("Analysis Period (UTC+0)")

    # Lista de periodos y opciones de intervalos permitidos por yfinance
    periods = ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max']
    interval_options = {
        '1d': ['1m','2m','5m','15m','30m','1h','1d'],
        '5d': ['1m','2m','5m','15m','30m','1h','1d'],
        '1mo': ['5m','15m','30m','1h','1d','1wk'],
        '3mo': ['15m','30m','1h','1d','1wk'],
        '6mo': ['1h','1d','1wk'],
        '1y': ['1h','1d','1wk','1mo'],
        '2y': ['1d','1wk','1mo'],
        '5y': ['1d','1wk','1mo'],
        '10y': ['1d','1wk','1mo'],
        'ytd': ['1h','1d','1wk'],
        'max': ['1d','1wk','1mo']
    }

    # Period selection
    selected_period = st.sidebar.selectbox("Select period", periods)
    
    # Interval selection based on the selected period
    if selected_period:
        allowed_intervals = interval_options[selected_period]
        selected_interval = st.sidebar.radio("Select interval", allowed_intervals, horizontal=True)

    st.sidebar.divider()

    st.sidebar.markdown('<a href="https://www.fisoft.es/" target="_blank" class="link-text">By: www.fisoft.es 🚀️</a>', unsafe_allow_html=True)

    return stock, selected_period, selected_interval, category, refresh_data

