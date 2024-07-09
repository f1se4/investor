import warnings
import streamlit as st
from streamlit_autorefresh import st_autorefresh

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

refresh_interval = 60000
st_autorefresh(interval=refresh_interval,key='datarefresh')

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Daily','Graphical Analysis', 
                                            'Information', 
                                            'ForeCasting',
                                            'Compound Interest Calculator'
                                            ])

    with tab2:
        ticker_data = information(stock, category)

    with tab1:# Crear un contenedor para los radiobuttons en horizontal
        options = ['1m','2m','5m','15m','90m','1h']
        radio_buttons = st.empty()
        
        # Generar el HTML y CSS para los radiobuttons en horizontal
        radio_html = """
        <div style="display: flex; align-items: center;">
          {buttons}
        </div>
        <script>
          // Añadir un event listener para actualizar el valor de los radiobuttons
          const radios = document.querySelectorAll('input[name="radio_horizontal"]');
          radios.forEach(radio => {
            radio.addEventListener('change', (event) => {
              const selectedValue = event.target.value;
              Streamlit.setComponentValue(selectedValue);
            });
          });
        </script>
        """
        
        # Crear los botones de radio en HTML
        buttons_html = ""
        for option in options:
            buttons_html += f"<label style="margin-right: 10px;"><input type="radio" name="radio_horizontal" value="{option}" style="margin-right: 5px;">{option}


                                        
            </label>
        # Insertar los botones de radio en el HTML
        radio_html = radio_html.format(buttons=buttons_html)
        # Mostrar los botones de radio en Streamlit
        radio_buttons.markdown(radio_html, unsafe_allow_html=True)        
        # Obtener el valor seleccionado (este valor se actualizará al seleccionar un radiobutton)
        selected_value = st.experimental_get_query_params().get("selected_radio", [None])[0]
        selected_interval = st.radio("Interval", options )
        placeholder = st.empty()
        with placeholder:
            daily(ticker_data, selected_interval)

    with tab3:
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

