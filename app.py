import warnings
import streamlit as st

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
    stock, start_time, end_time, category, subcategory = configure_sidebar()

    # Get data
    data = retrieve_data(stock, start_time, end_time)
    start_date_str = start_time.strftime("%Y-%m-%d")
    end_date_str = end_time.strftime("%Y-%m-%d")
    full_data = data.copy()
    data = data[start_date_str:end_date_str]

    # Render main title
    company_name = get_company_name(stock)
    st.markdown(f"""
        <div style="text-align: center; color: #2271B3; font-size: 28px; font-weight: bold; padding: 10px 0;">
                {stock} - {company_name}
        </div>
        <hr style="border: 2px solid #3B83BD;">
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Graphical Analysis', 
                                            'Information', 
                                            'Daily', 
                                            'ForeCasting',
                                            'Compound Interest Calculator'
                                            ])

    with tab2:
        ticker_data = information(stock, category)

    with tab3:
        daily(ticker_data)

    with tab1:
        analysis(data, full_data)
        
    with tab4:
        forecasting(data, end_time, start_time)

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

