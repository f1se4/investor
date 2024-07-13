import warnings
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

from calculations.calculations import (
    get_company_name,
    retrieve_data
)
from layout.sidebar          import configure_sidebar
from layout.analysis         import analysis
from layout.daily            import daily
from layout.forecasting      import forecasting
from layout.faqs             import faqs
from layout.information     import information
from layout.calculadora     import calculadora

# Suppress warnings for better display
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(layout="wide", page_title='FiserFinance Pro', page_icon='./assets/icon.jpg')


# Main function to run the app
def main():
    stock, selected_period, selected_interval, category, refresh_data = configure_sidebar()

    if refresh_data:
        refresh_interval = 60000
        st_autorefresh(interval=refresh_interval,key='datarefresh')

    # Get data
    # data = retrieve_data(stock, start_time, end_time)
    # start_date_str = start_time.strftime("%Y-%m-%d")
    # end_date_str = end_time.strftime("%Y-%m-%d")
    # full_data = data.copy()
    # data = data[start_date_str:end_date_str]

    # Render main title
    company_name = get_company_name(stock)
    st.markdown(f"""
        <div style="text-align: center; color: #2271B3; font-size: 28px; font-weight: bold; padding: 10px 0;">
                {stock} - {company_name}
        </div>
        <hr style="border: 2px solid #3B83BD;">
    """, unsafe_allow_html=True)

    data = yf.download(stock, period=selected_period, interval=selected_interval).drop(columns=['Adj Close'])

    if selected_period in ['1d','5d','1mo','3mo','6mo']:
        data_sm = yf.download(stock, period='1y', interval='1h').drop(columns=['Adj Close'])
    else:
        data_sm = data.copy()

    tab1, tab3, tab4, tab5 = st.tabs(['Analysis',
                                            'Information', 
                                            'ForeCasting',
                                            'Compound Interest Calculator'
                                            ])

    with tab3:
        information(stock, category)

    with tab1:# Crear un contenedor para los radiobuttons en horizontal
        # Añadir checkboxes para los indicadores
        col1, col2, col3 = st.columns(3)
        with col1:
            show_sma200 = st.checkbox('Show SMA200')
            show_sma5 = st.checkbox('Show SMA5')
        with col2:
            show_macd = st.checkbox('Show MACD')
            show_rsi = st.checkbox('Show RSI')
        with col3:
            show_volatility = st.checkbox('Show Volatility')
        placeholder = st.empty()
        with placeholder:
            daily(data, data_sm, show_sma200, show_sma5, show_macd, show_rsi, show_volatility)

    with tab4:
        pass
        # forcasting(data, end_time, start_time)

    with tab5:
        calculadora()

    st.divider()

    # Display raw data
    with st.expander('YFinance Raw Data'):
        st.dataframe(data)

    with st.expander('FAQs'):
        faqs()

if __name__ == '__main__':
    main()

