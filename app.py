import warnings
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

from calculations.calculations import get_company_name
from layout.sidebar          import configure_sidebar
from layout.faqs             import faqs
from layout.information     import information
from layout.calculadora     import calculadora
from visualizations.lightweight import f_daily_plot
import bot

# Suppress warnings for better display
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(layout="wide", page_title='FiserFinance Pro', page_icon='./assets/icon.jpg')


# Main function to run the app
def main():
    side_elements = [0,1,2,3,4,5]
    side_elements[0], side_elements[1], side_elements[2], side_elements[3], side_elements[4], side_elements[5] = configure_sidebar()

    if side_elements[0] == 'Analysis':
        stock = side_elements[1]
        selected_period = side_elements[2]
        selected_interval = side_elements[3]
        category = side_elements[4]
        refresh_data = side_elements[5]

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

        tab1, tab2 = st.tabs(['Analysis',
                              'Information', 
                              ])

        if len(data) == 0:
            st.write("Try another period/interval combination")
        else:
            with tab1:# Crear un contenedor para los radiobuttons en horizontal
                # Añadir checkboxes para los indicadores
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_sma200 = st.checkbox('SMA200')
                    show_sma5 = st.checkbox('SMA5', value=True)
                with col2:
                    show_macd = st.checkbox('MACD', value=True)
                    show_rsi = st.checkbox('RSI')
                with col3:
                    show_volatility = st.checkbox('Volatility')
                    show_bollinger  = st.checkbox('Bollinger Bands')
    
                placeholder = st.empty()
                with placeholder:
                    f_daily_plot(data, data_sm,
                             show_sma200, show_sma5, 
                             show_macd, show_rsi, 
                             show_volatility, show_bollinger)
    
        with tab2:
            information(stock, category)

        st.divider()
    
        # Display raw data
        with st.expander('YFinance Raw Data'):
            st.dataframe(data)
    
        with st.expander('FAQs'):
            faqs()
            # forcasting(data, end_time, start_time)

    if side_elements[0] == 'Trading':
        bot.bot_main()

    if side_elements[0] == 'Calculator':
        initial_investment = side_elements[1]
        monthly_contribution = side_elements[2]
        annual_interest_rate = side_elements[3]
        years = side_elements[4]
        calculadora(initial_investment, monthly_contribution, annual_interest_rate, years)
    

if __name__ == '__main__':
    main()

