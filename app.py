import datetime
from datetime import timedelta
import warnings
import streamlit as st
import pandas as pd
from PIL import Image
import yfinance as yf
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from calculations.calculations import (
    format_value, mostrar_informacion_general, arima_forecasting,
    get_company_name,daily_returns, returns_vol, retrieve_data
)
from visualizations.plotting import (
    plot_forecast_hw, plot_cmf_with_moving_averages,
    plot_with_indicators, plot_candlestick, plot_indicators,
    plot_volatility, plot_ma, plot_arima, plot_xgboost_forecast, plot_rendimiento,
    plot_price_and_volume
)
from layout.sidebar import configure_sidebar

# Suppress warnings for better display
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(layout="wide")

# Main function to run the app
def main():
    stock, start_time, end_time = configure_sidebar()

    # Get data
    data = retrieve_data(stock, start_time, end_time)

    # Render main title
    company_name = get_company_name(stock)
    st.markdown(f"""
        <div style="text-align: center; color: #4CAF50; font-size: 48px; font-weight: bold; padding: 10px 0;">
                {stock} - {company_name}
        </div>
        <hr style="border: 2px solid #4CAF50;">
    """, unsafe_allow_html=True)

    # Ticker data from yfinance
    ticker_data = yf.Ticker(stock)

    # Show general information
    st.subheader('Información general')
    with st.expander(""):
        mostrar_informacion_general(ticker_data)

    # Show performance plot
    st.subheader('Performance')
    rendimiento_plot = plot_rendimiento(stock)
    st.plotly_chart(rendimiento_plot, use_container_width=True)

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

    st.markdown("### Daily")
    col1, col2 = st.columns(2)
    with col1:
        plot_candlestick_3 = plot_candlestick(historical_data,False, False)
        st.plotly_chart(plot_candlestick_3)
    with col2:
        st.markdown(f"**Performance**: {format_value(price_change_percent)}%", unsafe_allow_html=True)
        st.markdown(f"**Δ Volume**: {format_value(volume_change)}", unsafe_allow_html=True)
        st.markdown(f"**Δ Openings**: {format_value(max_diff)}", unsafe_allow_html=True)
        st.markdown(f"**Δ Closing**: {format_value(min_diff)}", unsafe_allow_html=True)
        st.markdown(f"**Shadow Market**: {format_value(open_close)}", unsafe_allow_html=True)

    # Graphic Analysis section
    st.subheader('Graphic Analysis')
    selected_graph = st.radio("Graph Type", ['Line','Candle/Velas'])
    if selected_graph == 'Line':
        plot_full_fig = plot_price_and_volume(data)
        st.plotly_chart(plot_full_fig)
    elif selected_graph == 'Candle/Velas':
        plot_candlestick_fig = plot_candlestick(data)
        st.plotly_chart(plot_candlestick_fig)
        with st.expander("Patterns"):
            st.image(Image.open('assets/patterns.jpg'))

    st.markdown("### Indicators")

    # Crear columnas
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Agregar una casilla de verificación en cada columna
    with col1:
        checkbox1 = st.checkbox('Paddle Traders')
    
    with col2:
        checkbox2 = st.checkbox('RSI/MACD')
    
    with col3:
        checkbox3 = st.checkbox('CMF & SMA')

    with col4:
        checkbox4 = st.checkbox('Volatility & Daily Returns')

    # with col5:
    #     checkbox5 = st.checkbox('Opción 5')

    if checkbox1:
        # Plot synthetic indicators
        plot_ind_sintetico = plot_with_indicators(data)
        st.pyplot(plot_ind_sintetico)

    if checkbox2:
        plot_indicator = plot_indicators(data)
        st.pyplot(plot_indicator)
        with st.expander("Indicators"):
            st.markdown("""
            - **RSI (Relative Strength Index):**
              - El RSI es un indicador de momentum que mide la velocidad y el cambio de los movimientos de precios.
              - Se calcula como un valor oscilante entre 0 y 120 y se utiliza comúnmente para identificar condiciones de sobrecompra (por encima de 70) y sobreventa (por debajo de 30).
              - Cuando el RSI está por encima de 70, el activo se considera sobrecomprado, lo que podría indicar una posible corrección a la baja.
              - Por el contrario, cuando el RSI está por debajo de 30, el activo se considera sobrevendido, lo que podría señalar una oportunidad de compra.
            """)
            st.markdown("""
            - **MACD (Moving Average Convergence Divergence):**
                - El MACD es un indicador de seguimiento de tendencias que muestra la diferencia entre dos medias móviles exponenciales (EMA) de distintos períodos.
                - El MACD Line se representa como una línea sólida y la Signal Line como una línea punteada. Cuando la Línea MACD está por encima de la Línea de Señal, a menudo se colorea de verde.
                - Esto suele indicar un **impulso alcista** y es considerado un signo positivo para el activo, sugiriendo una tendencia alcista en el corto plazo.
                - Histograma del MACD: Además de las líneas, el MACD también se representa mediante un histograma que muestra la diferencia entre la Línea MACD y la Signal Line. En muchos gráficos, las barras del histograma se pintan de verde cuando la Línea MACD está por encima de la Signal Line, lo que refuerza la indicación de una tendencia alcista.
            """)

    if checkbox3:
        # Chaikin Money Flow & SMA
        st.subheader('Chaikin Money Flow & SMA')
        fig_cmf_ma = plot_cmf_with_moving_averages(data)
        st.pyplot(fig_cmf_ma)
        with st.expander('CMF'):
            st.markdown('''
        El **Chaikin Money** Flow (CMF) es un indicador técnico utilizado en el análisis financiero para medir la acumulación o distribución de un activo, basado en el volumen y el precio.

        El CMF puede ayudar a confirmar si la tendencia observada en las medias móviles está respaldada por el volumen. Por ejemplo, una media móvil que indica una tendencia alcista acompañada por un CMF positivo sugiere que la tendencia está respaldada por una presión de compra sostenida.

        - **Identificación de Divergencias:** Comparar el CMF con medias móviles puede ayudar a identificar divergencias entre el precio y el volumen. Si las medias móviles indican una tendencia alcista pero el CMF está en negativo, podría indicar que, a pesar de la subida del precio, hay una acumulación de presión de venta, lo que podría señalar una posible reversión de la tendencia.

        - **Validez de Señales de Compra/Venta:** Las medias móviles se utilizan a menudo para generar señales de compra y venta cuando cruzan ciertos niveles. El CMF puede validar estas señales. Por ejemplo, una señal de compra basada en un cruce de media móvil es más fiable si el CMF es positivo, ya que indica que la presión de compra está respaldando la subida del precio. Análisis de Sentimiento del Mercado:

        Las medias móviles reflejan la tendencia del precio a lo largo del tiempo. El CMF, al considerar el volumen, proporciona información sobre el sentimiento del mercado (acumulación o distribución). Comparar ambos puede ofrecer una visión más completa de la situación del mercado.
           ''')

    if checkbox4:
        # Daily Returns & Volatility
        st.subheader('Daily Returns & Volatility')
        df_ret = daily_returns(data)
        df_vol = returns_vol(df_ret)
        plot_vol = plot_volatility(df_vol)
        st.pyplot(plot_vol)
        with st.expander("Volatility"):
            st.markdown("""
            Este gráfico muestra los retornos diarios logarítmicos del precio de cierre del activo y la volatilidad asociada.
            - **Retornos Diarios:** Representan el cambio porcentual en el precio de cierre de un día a otro.
            - **Volatilidad:** Es la desviación estándar de los retornos diarios móviles (últimos 12 días), que indica la variabilidad en los cambios de precio.
            - La volatilidad alta puede indicar fluctuaciones significativas en el precio del activo.
            """)

    # Smoothing plot
    st.subheader('Smoothing')
    plot_ma_fig = plot_ma(data)
    st.pyplot(plot_ma_fig)

    # Forecasting section
    st.subheader('ForeCasting')
    if (end_time - start_time).days >= 35:
        col1, col2 = st.columns([1, 1])

        with col1:
            periods = st.number_input("Periods:", value=10, min_value=1, step=1)
        with col2:
            modelos = ["ARIMA", "Holt-Winters", "XGBoost"]
            modelo_seleccionado = st.radio("Forecasting Model:", modelos)


        if modelo_seleccionado == 'ARIMA':
            # Realizar forecasting con ARIMA
            forecast_arima = arima_forecasting(data, periods)
            arima_plot = plot_arima(data, forecast_arima, periods)
            st.pyplot(arima_plot)
            st.markdown("*ARIMA(3,1,3)*")
        elif modelo_seleccionado == 'Holt-Winters':
            # Crear el modelo Holt-Winters
            model = ExponentialSmoothing(data['Close'], trend='add', seasonal='add', seasonal_periods=12)
            fitted_model = model.fit()
            # Hacer la predicción para los próximos 5 días
            forecast = fitted_model.forecast(periods)
            st.write(f"Forecasting {periods} days for Model {modelo_seleccionado}")
            forecast_df = pd.DataFrame(forecast, columns=['forecast_values'])
            forecast_df['Date'] = pd.to_datetime(pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(forecast_df)))
            forecast_df.set_index('Date', inplace=True)
            # Mostrar gráficos de datos históricos y predicción
            forecast_plot = plot_forecast_hw(data, forecast_df)
            st.pyplot(forecast_plot)
        elif modelo_seleccionado == 'XGBoost':
            st.pyplot(plot_xgboost_forecast(data, periods))
    else:
        st.write("For an optimal forecasting you need at least 35 days")
    
    # Display raw data
    with st.expander('Raw Data'):
        st.dataframe(data)

if __name__ == '__main__':
    main()

