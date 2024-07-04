import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from datetime import datetime, timedelta
from PIL import Image
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from calculations.calculations import format_value, mostrar_informacion_general, arima_forecasting, get_company_name, get_data, daily_returns, returns_vol
from visualizations.plotting import plot_forecast_hw, plot_volume, plot_cmf_with_moving_averages, plot_with_indicators, plot_candlestick, plot_full, plot_indicators, plot_volatility, plot_ma, plot_arima, plot_xgboost_forecast, plot_rendimiento 

import warnings
# Suprimir advertencias para una mejor visualización
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

#########################################################################################
#### LAYOUT - Sidebar
#########################################################################################

logo = Image.open('assets/logo.jpeg')

# Leer los tickers desde el archivo
with open('tickers.txt', 'r') as file:
    tickers = [line.strip() for line in file.readlines()]

# Crear una lista de opciones con ticker y nombre de empresa
options = []
for ticker in tickers:
    company_name = get_company_name(ticker)
    options.append(f"{ticker} - {company_name}")

with st.sidebar:
    st.image(logo)
    selected_option = st.selectbox('Selecciona un Ticker', options, index=0)
    stock = selected_option.split(' - ')[0]
    st.header("")
    start_time = st.date_input(
                    "Fecha de Inicio",
                    datetime.today() - timedelta(days=182),
                    format="DD/MM/YYYY")
    end_time = st.date_input(
                    "Fecha Final",
                    datetime.today(),
                    format="DD/MM/YYYY")

    st.header("Help")
    st.image(Image.open('assets/velas.jpg'))

#########################################################################################
#### DATA - Funciones sobre inputs
#########################################################################################
data = get_data(stock, start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d"))

#########################################################################################
#### LAYOUT - Render Final
#########################################################################################

company_name = get_company_name(stock)
# Título de la app con estilos
st.markdown(f"""
    <div style="text-align: center; color: #4CAF50; font-size: 48px; font-weight: bold; padding: 10px 0;">
            {stock} - {company_name}
    </div>
    <hr style="border: 2px solid #4CAF50;">
""", unsafe_allow_html=True)

ticker_data = yf.Ticker(stock)

# Obtener información general del ticker
info = ticker_data.info

st.subheader('Información general')
with st.expander(""):
    # Mostrar información general
    mostrar_informacion_general(ticker_data)

st.subheader('Performance')
rendimiento_plot = plot_rendimiento(stock)
st.plotly_chart(rendimiento_plot, use_container_width=True)

# Obtener datos históricos
today = datetime.today().strftime('%Y-%m-%d')
five_days_ago = (datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')
historical_data = ticker_data.history(start=five_days_ago, end=today)

# Calcular KPIs para los últimos 2 días
last_day_data = historical_data.iloc[-1]
prev_day_data = historical_data.iloc[-2]

# Por ejemplo, calcular el cambio porcentual
price_change_percent = ((last_day_data['Close'] - prev_day_data['Close']) / prev_day_data['Close']) * 100
volume_change = last_day_data['Volume'] - prev_day_data['Volume']
max_diff = last_day_data['High'] - prev_day_data['High']
min_diff = last_day_data['Low'] - prev_day_data['Low']

st.markdown("### Daily")
# Crear dos columnas
col1, col2 = st.columns(2)
with col1:
    plot_candlestick_3 = plot_candlestick(historical_data)
    st.pyplot(plot_candlestick_3)
with col2:
    # Mostrar valores en Streamlit usando st.markdown
    st.markdown(f"**Performance**: {format_value(price_change_percent)}%", unsafe_allow_html=True)
    st.markdown(f"**Volume**: {format_value(volume_change)}", unsafe_allow_html=True)
    st.markdown(f"**Difference in Max Values**: {format_value(max_diff)}", unsafe_allow_html=True)
    st.markdown(f"**Difference in Min Values**: {format_value(min_diff)}", unsafe_allow_html=True)

st.subheader('Graphic Analysis')
plot_full_fig = plot_full(data)
st.pyplot(plot_full_fig)

plot_candlestick_fig = plot_candlestick(data)
st.pyplot(plot_candlestick_fig)
with st.expander("Patterns"):
    st.image(Image.open('assets/patterns.jpg'))

# Plotear volúmenes
plot_volume_fig = plot_volume(data)
st.pyplot(plot_volume_fig)
with st.expander("Volumen"):
    st.markdown("""
      - El volumen representa la cantidad total de acciones negociadas de un activo en un período de tiempo específico.
      - Se utiliza para evaluar la liquidez del mercado y la intensidad de las transacciones.
    """)

plot_ind_sintetico = plot_with_indicators(data)
st.pyplot(plot_ind_sintetico)

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
      El MACD Line se representa como una línea sólida y la Signal Line como una línea punteada. Cuando la Línea MACD está por encima de la Línea de Señal, 
      a menudo se colorea de verde. 
    Esto suele indicar un **impulso alcista** y es considerado un signo positivo para el activo, sugiriendo una tendencia alcista en el corto plazo.

        - Histograma del MACD: Además de las líneas, el MACD también se representa mediante un histograma que muestra la diferencia entre la Línea MACD y la Signal Line. En muchos gráficos, las barras del histograma se pintan de verde cuando la Línea MACD está por encima de la Signal Line, lo que refuerza la indicación de una tendencia alcista.
    """)

# Integrar en la interfaz de Streamlit
st.subheader('Chaikin Money Flow & SMA')

# Graficar CMF y Medias Móviles
fig_cmf_ma = plot_cmf_with_moving_averages(data)
st.pyplot(fig_cmf_ma)
with st.expander('CMF'):
    st.markdown('''
El **Chaikin Money** Flow (CMF) es un indicador técnico utilizado en el análisis financiero para medir la acumulación o distribución de un activo, basado en el volumen y el precio.

El CMF puede ayudar a confirmar si la tendencia observada en las medias móviles está respaldada por el volumen. Por ejemplo, una media móvil que indica una tendencia alcista acompañada por un CMF positivo sugiere que la tendencia está respaldada por una presión de compra sostenida.

- **Identificación de Divergencias:**
Comparar el CMF con medias móviles puede ayudar a identificar divergencias entre el precio y el volumen.
Si las medias móviles indican una tendencia alcista pero el CMF está en negativo, podría indicar que, a pesar de la subida del precio, hay una acumulación de presión de venta, lo que podría señalar una posible reversión de la tendencia.
- **Validez de Señales de Compra/Venta:** Las medias móviles se utilizan a menudo para generar señales de compra y venta cuando cruzan ciertos niveles.
El CMF puede validar estas señales. Por ejemplo, una señal de compra basada en un cruce de media móvil es más fiable si el CMF es positivo, ya que indica que la presión de compra está respaldando la subida del precio.
Análisis de Sentimiento del Mercado:

Las medias móviles reflejan la tendencia del precio a lo largo del tiempo.
El CMF, al considerar el volumen, proporciona información sobre el sentimiento del mercado (acumulación o distribución). Comparar ambos puede ofrecer una visión más completa de la situación del mercado.
   ''')

# Renderizar gráfico de medias móviles
st.subheader('Smoothing')
plot_ma_fig = plot_ma(data)
st.pyplot(plot_ma_fig)

st.subheader('Daily Returns & Volatitlity')
df_ret = daily_returns(data)
df_vol = returns_vol(df_ret)
plot_vol = plot_volatility(df_vol)
st.pyplot(plot_vol)
with st.expander("Volatility"):
    # Explicación de los retornos diarios y volatilidad
    st.markdown("""
    Este gráfico muestra los retornos diarios logarítmicos del precio de cierre del activo y la volatilidad asociada.
    - **Retornos Diarios:** Representan el cambio porcentual en el precio de cierre de un día a otro.
    - **Volatilidad:** Es la desviación estándar de los retornos diarios móviles (últimos 12 días), que indica la variabilidad en los cambios de precio.     - La volatilidad alta puede indicar fluctuaciones significativas en el precio del activo.
    """)

# Mostrar datos históricos y predicción en la interfaz
st.subheader('ForeCasting')

if (end_time - start_time).days >= 35:
    
    # Configurar dos columnas en Streamlit
    col1, col2 = st.columns([1, 1])
    
    # Columna 1: number_input para los períodos
    with col1:
        periods = st.number_input("Periods:", value=10, min_value=1, step=1)
    
    # Columna 2: st.radio para seleccionar el modelo de forecasting
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
    
st.subheader('Raw Data')
st.dataframe(data)
