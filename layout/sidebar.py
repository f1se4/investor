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

    st.sidebar.header("Analysis Period")
    start_time = st.sidebar.date_input("Fecha de Inicio",
                                       datetime.datetime.today() - timedelta(days=182),
                                       format="DD/MM/YYYY")
    end_time = st.sidebar.date_input("Fecha Final",
                                     datetime.datetime.today(),
                                     format="DD/MM/YYYY")
    st.sidebar.header("Help")
    st.sidebar.image(Image.open('assets/velas.jpg'))

    st.sidebar.markdown('<a href="https://www.fisoft.es/" target="_blank" class="link-text">By: www.fisoft.es</a>', unsafe_allow_html=True)

    return stock, start_time, end_time, category, subcategory

