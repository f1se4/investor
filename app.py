import warnings
import streamlit as st
import yfinance as yf

from calculations.calculations import (
    mostrar_informacion_general,
    get_company_name,
    obtener_constituyentes_y_pesos, retrieve_data
)
from visualizations.plotting import plot_rendimiento
from layout.sidebar          import configure_sidebar
from layout.analysis         import analysis
from layout.daily            import daily
from layout.forecasting      import forecasting
from layout.faqs             import faqs
from layout.information      import obtener_pesos_constrituyentes

# Suppress warnings for better display
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(layout="wide", page_title='FiserFinance Pro', page_icon='./assets/icon.jpg')

# Main function to run the app
def main():
    stock, start_time, end_time, category, subcategory = configure_sidebar()

    # Get data
    data = retrieve_data(stock, start_time, end_time)

    # Render main title
    company_name = get_company_name(stock)
    st.markdown(f"""
        <div style="text-align: center; color: #2271B3; font-size: 28px; font-weight: bold; padding: 10px 0;">
                {stock} - {company_name}
        </div>
        <hr style="border: 2px solid #3B83BD;">
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(['Graphical Analysis', 'Information', 'Daily', 'ForeCasting'])

    with tab2:
        # Show performance plot
        st.subheader('Performance')
        rendimiento_plot = plot_rendimiento(stock)
        st.plotly_chart(rendimiento_plot, use_container_width=True)
        # Ticker data from yfinance
        ticker_data = yf.Ticker(stock)

        # Show general information
        st.subheader('General Information')
        mostrar_informacion_general(ticker_data)

        if category == 'Indexados':
            st.subheader('Constituents')
            st.dataframe(obtener_constituyentes_y_pesos(stock))


    with tab3:
        daily(ticker_data)

    with tab1:
        analysis(data)
        
    with tab4:
        forecasting(data, end_time, start_time)

    st.divider()

    # Display raw data
    with st.expander('YFinance Raw Data'):
        st.dataframe(data)

    with st.expander('FAQs'):
        faqs()

if __name__ == '__main__':
    main()

