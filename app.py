import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mplfinance as mpf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#from alpha_vantage.timeseries import TimeSeries

# # Librería tiempo real
# import finnhub
# api_key = 'cq266u1r01ql95ncj820cq266u1r01ql95ncj82g'
# finnhub_client = finnhub.Client(api_key=api_key)

#API_KEY = 'TVOT6OCQEK98JBTR'

st.set_page_config(layout="wide")

# Configuración global de tamaño de fuente para matplotlib
mpl.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.size': 10})

plt.style.use("dark_background")
###########################
#### Funciones Principales
###########################
# Función para realizar forecasting con ARIMA
def arima_forecasting(data, periods):
    # Ajustar el modelo ARIMA automáticamente
    model = ARIMA(data, order=(1, 1, 1))  # Ejemplo con ARIMA(1,1,1)
    fitted_model = model.fit()
    
    # Hacer las predicciones
    forecast = fitted_model.forecast(steps=periods)
    
    return forecast

# Función para realizar forecasting con auto_arima
def auto_arima_forecasting(data, periods):
    from pmdarima.arima import auto_arima
    model = auto_arima(data, seasonal=False, suppress_warnings=True)
    forecast = model.predict(n_periods=periods)
    
    return forecast
# Función para obtener los datos en tiempo real desde Alpha Vantage
# def get_realtime_data(symbol, interval='1min', outputsize='compact'):
#     try:
#         ts = TimeSeries(key=API_KEY, output_format='pandas')
#         data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize=outputsize)
#         return data
#     except ValueError as e:
#         st.error(f"Error al obtener datos: {e}")
#         return None
#
# # Función para plotear el gráfico en tiempo real
# def plot_realtime_graph(symbol):
#     data = get_realtime_data(symbol)
#     if data is not None:
#         fig, ax = plt.subplots(figsize=(14, 6))
#         ax.plot(data.index, data['4. close'], label='Close Price', color='dodgerblue')
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Price')
#         ax.set_title(f'Intraday Price Chart for {symbol}')
#         ax.legend()
#         st.pyplot(fig)
#
# # Función para plotear gráfico en tiempo real
# def plot_realtime_graph(symbol):
#     res = get_realtime_data(symbol)
#     timestamps = [datetime.datetime.fromtimestamp(ts) for ts in res['t']]
#     close_prices = res['c']
#
#     plt.figure(figsize=(12, 6))
#     plt.plot(timestamps, close_prices, label=f'Precio de cierre - {symbol}')
#     plt.xlabel('Tiempo')
#     plt.ylabel('Precio')
#     plt.title(f'Gráfico de precios en tiempo real para {symbol}')
#     plt.legend()
#     plt.grid(True)
#     st.pyplot()
#
def forecast_next_days(data, num_days=5):
    # Utilizar la columna 'Close' para el modelo AR
    close_series = data['Close']

    # Entrenar el modelo AR con un retraso de 1 día (AR(1))
    model = AutoReg(close_series, lags=1)
    model_fit = model.fit()

    # Predecir los próximos num_days días
    forecast_values = model_fit.predict(start=len(close_series), end=len(close_series)+num_days-1, dynamic=False)

    # Crear índice de fechas para las predicciones
    forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=num_days, freq='D')

    # Crear un DataFrame para las predicciones
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_values})

    return forecast_df

def plot_forecast_hw(data, forecast):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Gráfico del Precio Histórico
    ax.plot(data.index, data.Close, color='dodgerblue', linewidth=1)

    # Gráfico de la Predicción
    ax.plot(forecast.index, forecast, label='ForeCast', color='orange', linestyle='--', linewidth=1)

    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=45, which='both')
    ax.grid(True,color='gray', linestyle='-', linewidth=0.01)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks([])  # Quitar los ticks
    
    ax.legend()
    fig.tight_layout()

    return fig

def plot_volume(data):
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(data.index, data['Volume'], color='dodgerblue', alpha=0.7)
    ax.set_ylabel('')
    # ax.grid(True, color='gray', linestyle='-', linewidth=0.2)
    ax.yaxis.set_ticks([])  # Quitar los ticks
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    return fig

# # Función para obtener datos en tiempo real desde Finnhub
# def get_realtime_data(symbol, resolution='1', count=50):
#     # Obtener la fecha y hora actual
#     to_time = int(datetime.datetime.now().timestamp())
#     # Calcular el tiempo inicial restando 'count' veces 'resolution' segundos a la fecha y hora actual
#     from_time = to_time - count * int(resolution)
#     res = finnhub_client.stock_candles(symbol=symbol, resolution=resolution, _from=from_time, to=to_time)
#
#     return res

def get_company_name(ticker):
    # Crear un objeto ticker con yfinance
    ticker_data = yf.Ticker(ticker)
    
    # Obtener el nombre de la empresa
    company_name = ticker_data.info['longName']
    
    return company_name

def get_data(stock, start_time, end_time):
    df = yf.download(stock, start=start_time, end=end_time)
    return df

def normalize_sma(data, sma, window):
    sma_mean = sma.rolling(window=window).mean()
    sma_std = sma.rolling(window=window).std()
    normalized_sma = (sma - sma_mean) / sma_std
    return normalized_sma

def calculate_cmf(data, period=8):
    mfv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']) * data['Volume']
    cmf = mfv.rolling(window=period).sum() / data['Volume'].rolling(window=period).sum()
    return cmf

def normalize_to_range(data, min_val, max_val):
    return 2 * ((data - min_val) / (max_val - min_val)) - 1

def calculate_moving_average(data, window):
    return data['Close'].rolling(window=window).mean()

def normalize_sma_to_range(data, sma, window, min_val=-1, max_val=1):
    sma_mean = sma.rolling(window=window).mean()
    sma_std = sma.rolling(window=window).std()
    normalized_sma = (sma - sma_mean) / sma_std
    
    # Normalizar al rango [-1, 1]
    return normalize_to_range(normalized_sma, normalized_sma.min(), normalized_sma.max())

def normalize_cmf_to_range(data, period=8, min_val=-1, max_val=1):
    cmf = calculate_cmf(data, period)
    return normalize_to_range(cmf, cmf.min(), cmf.max())

def plot_cmf_with_moving_averages(data, cmf_period=8, ma_period1=5, ma_period2=20):
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Calcular CMF
    cmf = calculate_cmf(data, period=cmf_period)
    norm_cmf = normalize_cmf_to_range(data, period=cmf_period)
    
    # Calcular Medias Móviles
    ma1 = calculate_moving_average(data, window=ma_period1)
    ma2 = calculate_moving_average(data, window=ma_period2)

    # Normalizar Medias Móviles al rango [0, 1]
    norm_ma1 = normalize_sma_to_range(data, ma1, ma_period1)
    norm_ma2 = normalize_sma_to_range(data, ma2, ma_period2)

    
    # Graficar CMF
    ax.bar(data.index, norm_cmf, width=1.5, color=np.where(cmf >= 0, 'green', 'red'), alpha=0.3, label ='CFD(8)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    
    # Graficar Medias Móviles
    # ax.plot(data.index, ma1, label=f'SMA({ma_period1})', color='dodgerblue')
    # ax.plot(data.index, ma2, label=f'SMA({ma_period2})', color='green')

    # Graficar Medias Móviles Normalizadas
    ax.plot(data.index, norm_ma1, label=f'SMA {ma_period1}', color='dodgerblue')
    ax.plot(data.index, norm_ma2, label=f'SMA {ma_period2}', color='rosybrown')
    
    # Personalizar el gráfico
    ax.legend(loc='best')
    ax.grid(True, color='gray', linestyle='-', linewidth=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks([])  # Quitar los ticks
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    
    return fig


def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 120 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, short_window=12, long_window=26):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def plot_candlestick(data):
    fig, ax = plt.subplots(figsize=(14, 6))
    mpf.plot(data,type='candle', style='yahoo',ax=ax, ylabel='')

    # Mover el eje y del precio al lado izquierdo
    ax.yaxis.tick_left()

    # Configurar la posición de la etiqueta del eje y del precio
    ax.yaxis.set_label_position('left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Quitar los ticks y etiquetas del eje y izquierdo
    ax.yaxis.set_ticks([])  # Quitar los ticks
    ax.set_yticklabels([])  # Quitar las etiquetas de los ticks

    ax.grid(True, color='gray', linestyle='-', linewidth=0.01)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # Ajustar el margen derecho para mostrar toda la información de la fecha
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig

def plot_full(data):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Gráfico del Precio
    ax.plot(data.index, data.Close, color='dodgerblue', linewidth=1)
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=45, which='both')
    ax.grid(True,color='gray', linestyle='-', linewidth=0.01)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.yaxis.set_ticks([])  # Quitar los ticks

    # Añadir texto de máximo y mínimo
    max_price = data['Close'].max()
    min_price = data['Close'].min()
    
    # Encontrar el índice correspondiente al máximo y mínimo
    idx_max = data['Close'].idxmax()
    idx_min = data['Close'].idxmin()
    
    plt.text(idx_max, max_price, f'{max_price:.3f}', va='bottom', ha='center', color='dodgerblue')
    plt.text(idx_min, min_price, f'{min_price:.3f}', va='top', ha='center', color='dodgerblue')

    fig.tight_layout()

    return fig

def plot_indicators(data):
    plt.rcParams.update({'font.size': 10})
    levels = get_levels(data)
    df_levels = pd.DataFrame(levels, columns=['index','close'])
    df_levels.set_index('index', inplace=True)

    fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True)

    # Plot RSI
    rsi = calculate_rsi(data)
    axs[0].plot(data.index, rsi, color='orange', linewidth=1)
    axs[0].axhline(70, color='red', linestyle='--', linewidth=0.7)
    axs[0].axhline(30, color='green', linestyle='--', linewidth=0.7)
    axs[0].set_ylabel('RSI')
    axs[0].tick_params(axis='x', rotation=45, which='both')
    axs[0].grid(True, color='gray', linestyle='-', linewidth=0.01)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].yaxis.set_ticks([])  # Quitar los ticks
    axs[0].xaxis.set_ticks([])  # Quitar los ticks

    # Plot MACD
    macd_line, signal_line = calculate_macd(data)
    axs[1].plot(data.index, macd_line, label='MACD', color='dodgerblue', linewidth=1)
    axs[1].plot(data.index, signal_line, label='Signal', color='magenta', linewidth=1, linestyle='--')
    axs[1].fill_between(data.index, macd_line, signal_line, where=(macd_line >= signal_line), color='green', alpha=0.3)
    
    # Plot Histograma del MACD
    macd_histogram = macd_line - signal_line
    axs[1].bar(data.index, macd_histogram, color=np.where(macd_histogram >= 0, 'green', 'darkgray'), alpha=0.6)

    axs[1].axhline(0, color='gray', linestyle='-', linewidth=0.5)
    axs[1].set_ylabel('MACD')
    axs[1].tick_params(axis='x', rotation=45, which='both')
    axs[1].grid(True, color='gray', linestyle='-', linewidth=0.01)
    axs[1].spines['top'].set_visible(False)
    axs[1].yaxis.set_ticks([])  # Quitar los ticks

    fig.tight_layout()

    return fig

def get_levels(dfvar):
    def isSupport(df,i):
        support = df['low'][i] < df['low'][i-1]  and df['low'][i] < df['low'][i+1] and df['low'][i+1] < df['low'][i+2] and df['low'][i-1] < df['low'][i-2]
        return support

    def isResistance(df,i):
        resistance = df['high'][i] > df['high'][i-1]  and df['high'][i] > df['high'][i+1] and df['high'][i+1] > df['high'][i+2] and df['high'][i-1] > df['high'][i-2]
        return resistance

    def isFarFromLevel(l, levels, s):
        level = np.sum([abs(l-x[0]) < s  for x in levels])
        return  level == 0
    
    
    df = dfvar.copy()
    df.rename(columns={'High':'high','Low':'low'}, inplace=True)
    s =  np.mean(df['high'] - df['low'])
    levels = []
    for i in range(2,df.shape[0]-2):
        if isSupport(df,i):  
            levels.append((i,df['low'][i]))
        elif isResistance(df,i):
            levels.append((i,df['high'][i]))

    filter_levels = []
    for i in range(2,df.shape[0]-2):
        if isSupport(df,i):
            l = df['low'][i]
            if isFarFromLevel(l, levels, s):
                filter_levels.append((i,l))
        elif isResistance(df,i):
            l = df['high'][i]
            if isFarFromLevel(l, levels, s):
                filter_levels.append((i,l))

    return filter_levels

def daily_returns(df):
    df = df.sort_index(ascending=True)
    df['returns'] = np.log(df['Close']).diff()
    return df

def returns_vol(df):
    df['volatility'] = df.returns.rolling(12).std()
    return df

def plot_volatility(df_vol):
    df_plot = df_vol.copy()
    fig = plt.figure(figsize=(12,6))
    plt.plot(df_plot.index, df_plot.returns, color='dodgerblue', linewidth=0.5, alpha=0.6)
    plt.plot(df_plot.index, df_plot.volatility, color='darkorange', linewidth=1)
    plt.ylabel('')
    plt.xticks(rotation=45,  ha='right')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.2)
    ax.axhline(0.01, color='red', linestyle='--', alpha=0.8)
    ax.axhline(0.005, color='green', linestyle='--', alpha=0.8)
    ax.yaxis.set_ticks([])  # Quitar los ticks
    plt.grid(True,color='gray', linestyle='-', linewidth=0.01)
    plt.legend(('Returns', 'Volatility'), frameon=False)

    # Añadir texto de máximo y mínimo
    max_vol = df_plot['volatility'].max()
    min_vol = df_plot['volatility'].min()
    
    # Encontrar el índice correspondiente al máximo y mínimo
    idx_max = df_plot['volatility'].idxmax()
    idx_min = df_plot['volatility'].idxmin()
    
    plt.text(idx_max, max_vol, f'{max_vol:.3f}', va='bottom', ha='center', color='darkorange', fontsize=10)
    plt.text(idx_min, min_vol, f'{min_vol:.3f}', va='top', ha='center', color='darkorange', fontsize=10)
    fig.tight_layout()

    return fig


def plot_ma(data):
    # Calcular medias móviles
    fig, ax = plt.subplots(figsize=(12, 6))
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()

    # Calcular media móvil exponencial (EMA)
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()

    # Calcular Holt-Winters - Aditivo
    try:
        model = ExponentialSmoothing(data['Close'], trend='add', seasonal='add', seasonal_periods=12)
        fitted = model.fit()
        data['Holt_Winters'] = fitted.fittedvalues
        ax.plot(data.index, data['Holt_Winters'], label='Holt-Winters', color='magenta', linestyle='--', linewidth=1)
    except:
        print('error')

    # Configurar gráfico
    ax.plot(data.index, data['Close'], label='Close', color='dodgerblue', linewidth=1, alpha=0.3)
    ax.plot(data.index, data['MA_20'], label='MA 20', color='orange', linestyle='--', linewidth=1)
    ax.plot(data.index, data['MA_50'], label='MA 50', color='green', linestyle='--', linewidth=1)
    ax.plot(data.index, data['EMA_10'], label='EMA_10', color='red', linestyle='--', linewidth=1)

    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.yaxis.set_ticks([])  # Quitar los ticks
    ax.grid(True, color='gray', linestyle='-', linewidth=0.01)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()

    return fig

def plot_arima(data, forecast_arima, forecast_periods):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], color='dodgerblue', linewidth=1, alpha=0.8)
    ax.plot(pd.date_range(start=data.index[-1], periods=forecast_periods+1, freq='D')[1:], forecast_arima, label='Forecast', linestyle='--', color='orange')
    ax.legend()
    ax.yaxis.set_ticks([])  # Quitar los ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(True, color='gray', linestyle='-', linewidth=0.01)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    return fig

###########################
#### LAYOUT - Sidebar
###########################

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
    start_time = st.date_input(
                    "Fecha de Inicio",
                    datetime.date.today() - datetime.timedelta(days=182),
                    format="DD/MM/YYYY")
    end_time = st.date_input(
                    "Fecha Final",
                    datetime.date.today(),
                    format="DD/MM/YYYY")

    st.header("Help")
    st.image(Image.open('assets/velas.jpg'))

###########################
#### DATA - Funciones sobre inputs
###########################

data = get_data(stock, start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d"))


###########################
#### LAYOUT - Render Final
###########################

company_name = get_company_name(stock)
st.title(f"{stock} - {company_name}")

plot_full_fig = plot_full(data)
st.pyplot(plot_full_fig)

plot_candlestick_fig = plot_candlestick(data)
st.pyplot(plot_candlestick_fig)

# Plotear volúmenes
plot_volume_fig = plot_volume(data)
st.pyplot(plot_volume_fig)
with st.expander("Volumen"):
    st.markdown("""
      - El volumen representa la cantidad total de acciones negociadas de un activo en un período de tiempo específico.
      - Se utiliza para evaluar la liquidez del mercado y la intensidad de las transacciones.
    """)


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
st.subheader('Chaikin Money Flow y Medias Móviles')

# Graficar CMF y Medias Móviles
fig_cmf_ma = plot_cmf_with_moving_averages(data)
st.pyplot(fig_cmf_ma)

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

# Configurar dos columnas en Streamlit
col1, col2 = st.columns([1, 1])

# Columna 1: number_input para los períodos
with col1:
    periods = st.number_input("Periods:", value=10, min_value=1, step=1)

# Columna 2: st.radio para seleccionar el modelo de forecasting
with col2:
    modelos = ["ARIMA", "Holt-Winters", "LSTM"]
    modelo_seleccionado = st.radio("Forecasting Model:", modelos)

if modelo_seleccionado == 'ARIMA':
    # Realizar forecasting con ARIMA
    forecast_arima = arima_forecasting(data['Close'], periods)
    arima_plot = plot_arima(data, forecast_arima, periods)
    st.pyplot(arima_plot)
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
    st.dataframe(forecast_df)

st.subheader('Raw Data')
st.dataframe(data)
