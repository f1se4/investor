import datetime
from datetime import timedelta
import streamlit as st
from PIL import Image
import json

from calculations.calculations import get_company_name

# Lista de periodos y opciones de intervalos permitidos por yfinance
periods = ['1d','5d','1mo','3mo','1y','2y','5y','10y','ytd','max']
interval_options = {
            '1d': ['1m','2m','5m','15m','30m','1h','1d'],
            '5d': ['1m','2m','5m','15m','30m','1h','1d'],
            '1mo': ['5m','15m','30m','1h','1d','1wk'],
            '3mo': ['30m','1h','1d','1wk'],
            '1y': ['1h','1d','1wk','1mo'],
            '2y': ['1d','1wk','1mo'],
            '5y': ['1d','1wk','1mo'],
            '10y': ['1d','1wk','1mo'],
            'ytd': ['1h','1d','1wk'],
            'max': ['1d','1wk','1mo']
        }

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
    st.sidebar.header("Application")
    selected_tab = st.sidebar.radio("", ["Analysis", "Trading", "Calculator"], horizontal=True)
    # selected_tab = st.sidebar.radio("", ["Analysis", "Calculator"], horizontal=True)
    
    if selected_tab == 'Analysis':
    # with sd_tab1:
        st.sidebar.markdown("## Select Ticker")
    
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
    
    
        # Period selection
        selected_period = st.sidebar.selectbox("Select period", periods, index=1)
        
        # Interval selection based on the selected period
        if selected_period:
            allowed_intervals = interval_options[selected_period]
            selected_interval = st.sidebar.radio("Select interval", allowed_intervals, horizontal=True)

    if selected_tab == 'Calculator':
        initial_investment = st.sidebar.number_input("Initial Investment (€)", min_value=0.0, value=1000.0, step=100.0)
        monthly_contribution = st.sidebar.number_input("Monthly Contribution (€)", min_value=0.0, value=100.0, step=10.0)
        annual_interest_rate = st.sidebar.number_input("Annual Interest Rate (%)", min_value=0.0, value=5.0, step=0.1)
        years = st.sidebar.number_input("Number of Years", min_value=1, value=10, step=1)

    if selected_tab == 'Trading':
        refresh_data = st.sidebar.checkbox('Refresh Data (60s)', value=True)
        st.sidebar.markdown("## Trade Indicators")
        select_g_strategy = st.sidebar.checkbox('Volume Profile', value=False)
        show_par = st.sidebar.checkbox('Parabolic SAR', value=False)
        select_trade_simple = st.sidebar.checkbox('Bollinger Bands', value=False)
        select_MMI = st.sidebar.checkbox('SMA I', value=False)
        select_MM = st.sidebar.checkbox('SMA II', value=False)
        st.sidebar.subheader("Indicators Parameters")
        if select_MMI:
            smai = st.sidebar.number_input(label='SMA I Window',value=200, step=10)
        else:
            smai = 1
        if select_MM:
            smaii = st.sidebar.number_input(label='SMA II Window',value=100, step=10)
        else:
            smaii = 1
        st.sidebar.divider()
        select_period_trade = st.sidebar.selectbox("Select period", periods)
        # Interval selection based on the selected period
        if select_period_trade:
            allowed_intervals = interval_options[select_period_trade]
            selected_interval_trading = st.sidebar.radio("Select interval", allowed_intervals, horizontal=True)
        st.sidebar.divider()
        values = st.sidebar.slider("Select a range of values for display", 1, 1440, 100 )

    
    st.sidebar.divider()

    st.sidebar.markdown('<a href="https://www.fisoft.es/" target="_blank" class="link-text">By: www.fisoft.es 🚀️</a>', unsafe_allow_html=True)

    if selected_tab == 'Analysis':
        return (selected_tab, stock, selected_period, selected_interval, category, refresh_data, 
               'NONE', 'NONE', 'NONE', 'NONE', 'NONE','NONE')
    elif selected_tab == 'Calculator':
        return (selected_tab, initial_investment, monthly_contribution, annual_interest_rate, years, 
               'NONE', 'NONE', 'NONE', 'NONE', 'NONE','NONE','NONE')
    elif selected_tab == 'Trading':
        return (selected_tab, selected_interval_trading, select_g_strategy, 
               select_trade_simple, refresh_data, select_MM, values, 
               select_period_trade, smai, smaii, select_MMI, show_par)


