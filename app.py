import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from datetime import datetime, timedelta
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mplfinance as mpf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import seaborn as sns
# Configuración de estilo para gráficos
#sns.set(style="whitegrid")

st.set_page_config(layout="wide")

# Configuración global de tamaño de fuente para matplotlib
mpl.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.size': 10})

plt.style.use("dark_background")

#########################################################################################
#### Funciones Cálculos
#########################################################################################
def mostrar_informacion_general(ticker_data):
    info = {
        "Nombre": ticker_data.info.get('longName', 'N/A'),
        "Símbolo": ticker_data.info.get('symbol', 'N/A'),
        "Tipo": ticker_data.info.get('quoteType', 'N/A'),
        "Sector": ticker_data.info.get('sector', 'N/A'),
        "Exchange": ticker_data.info.get('exchange', 'N/A'),
        "Divisa": ticker_data.info.get('currency', 'N/A'),
        "Precio anterior de cierre": ticker_data.info.get('previousClose', 'N/A'),
        "Precio de apertura": ticker_data.info.get('open', 'N/A'),
        "Precio más bajo del día": ticker_data.info.get('dayLow', 'N/A'),
        "Precio más alto del día": ticker_data.info.get('dayHigh', 'N/A'),
        "Volumen promedio (10 días)": ticker_data.info.get('averageVolume10days', 'N/A'),
        "Volumen": ticker_data.info.get('volume', 'N/A'),
        "Ratio P/E (trailing)": ticker_data.info.get('trailingPE', 'N/A'),
        "Ratio PEG (trailing)": ticker_data.info.get('trailingPegRatio', 'N/A'),
        "Rango de 52 semanas - Mínimo": ticker_data.info.get('fiftyTwoWeekLow', 'N/A'),
        "Rango de 52 semanas - Máximo": ticker_data.info.get('fiftyTwoWeekHigh', 'N/A')
    }
    
    for key, value in info.items():
        st.write(f"**{key}:** {value}")

# Función para calcular la Repulsión Alisada (similar a una EMA)
def repulsion_alisada(data, span):
    return data.ewm(span=span, adjust=False).mean()

# Función para normalizar una serie
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Función para calcular la TEMA
def tema(data, window):
    ema1 = data.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3

# Función para calcular la DEMA
def dema(data, window):
    ema1 = data.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    return 2 * ema1 - ema2

# Función para realizar forecasting con ARIMA
def arima_forecasting(data, periods):
    # Ajustar el modelo ARIMA automáticamente
    model = ARIMA(data, order=(1, 1, 1))  # Ejemplo con ARIMA(1,1,1)
    fitted_model = model.fit()
    
    # Hacer las predicciones
    forecast = fitted_model.forecast(steps=periods)
    
    return forecast

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

#########################################################################################
#### Funciones para gráficos
#########################################################################################

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

def plot_with_indicators(data):
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Graficar el precio de cierre
    #ax.plot(data.index, data['Close'], label='Precio de Cierre', color='blue', alpha=0.5)
    
    # Calcular y graficar la Repulsión Alisada
    repulsion = repulsion_alisada(data['Close'], span=5)
    #repulsion_raw = repulsion_alisada(data['Close'], span=5)
    #repulsion = normalize(repulsion_raw)
    ax.plot(data.index, repulsion, label='Repulsión Alisada (5)', color='dodgerblue', linestyle='-', alpha=0.8)
    
    # Calcular y graficar la TEMA
    #tema_line_raw = tema(data['Close'], window=21)
    tema_line = tema(data['Close'], window=21)
    #tema_line = normalize(tema_line_raw)
    ax.plot(data.index, tema_line, label='TEMA (21)', color='orange', linestyle='--', alpha=0.8)
    
    # Calcular y graficar la DEMA
    #dema_line_raw = dema(data['Close'], window=21)
    dema_line = dema(data['Close'], window=21)
    #dema_line = normalize(dema_line_raw)
    ax.plot(data.index, dema_line, label='DEMA (21)', color='r', linestyle='--', alpha=0.8)

    # Rellenar el área según las condiciones
    ax.fill_between(data.index, repulsion, tema_line, 
                    where=(repulsion > tema_line) & (repulsion> dema_line), 
                    color='blue', alpha=0.3, interpolate=True)
    
    ax.fill_between(data.index, repulsion, tema_line, 
                    where=(repulsion > tema_line) & (repulsion <= dema_line), 
                    color='green', alpha=0.3, interpolate=True)
    
    # Configuraciones del gráfico
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True,color='gray', linestyle='-', linewidth=0.01)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks([])  # Quitar los ticks
    fig.tight_layout()
    return fig


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

def mostrar_grafico_precios(historical_data):
    # Crear figura y ejes para el gráfico
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficar precios de cierre
    ax.plot(historical_data.index, historical_data['Close'], marker='o', linestyle='-', color='b', label='Precio de cierre')

    # Añadir etiquetas y título
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio de cierre')
    ax.set_title('Precio de cierre en los últimos 2 días')
    ax.legend()

    # Mostrar el gráfico en Streamlit
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
    ax.plot(data.index, data['Close'], label='Real', color='dodgerblue', linewidth=1, alpha=0.3)
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


def plot_rendimiento(ticker):
    # Función para obtener los datos de rendimiento
    def get_performance_data(ticker, period):
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data
    
    # Obtener datos de rendimiento para ticker de ejemplo 'AAPL'
    periods = ['1wk', '1mo', '6mo', '1y']
    data = []
    
    # Obtener datos y manejar casos sin datos disponibles
    for period in periods:
        try:
            data.append(get_performance_data(ticker, period))
        except:
            data.append(None)
    
    # Crear una fila para los gráficos de velocímetro
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={'projection': 'polar'})
    
    # Rangos máximos para los gráficos de velocímetro
    ranges = {
        '1wk': (-10, 10),
        '1mo': (-20, 20),
        '6mo': (-50, 50),
        '1y': (-100, 100)
    }
    
    # Generar los gráficos de velocímetro
    for i, period in enumerate(periods):
        min_val, max_val = ranges[period]
        if data[i] is not None and not data[i].empty:
            performance = data[i]['Close'].pct_change().iloc[-1] * 100
            plot_velocimeter(axs[i], performance, min_val, max_val, period)
        else:
            axs[i].set_title(f'No data ({period})', fontsize=12)
            axs[i].set_axis_off()

    return fig

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
st.title(f"{stock} - {company_name}")

ticker_data = yf.Ticker(stock)

# Obtener información general del ticker
info = ticker_data.info

st.subheader('Información general')
with st.expander(""):
    # Mostrar información general
    mostrar_informacion_general(ticker_data)

# Obtener datos históricos
today = datetime.today().strftime('%Y-%m-%d')
two_days_ago = (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d')
historical_data = ticker_data.history(start=two_days_ago, end=today)

rendimiento_plot = plot_rendimiento(stock)
st.pyplot(rendimiento_plot)

# Mostrar datos históricos
st.subheader('Datos históricos de los últimos 2 días')
st.write(historical_data)

# Calcular KPIs para los últimos 2 días
last_day_data = historical_data.iloc[-1]
prev_day_data = historical_data.iloc[-2]

# Por ejemplo, calcular el cambio porcentual
price_change_percent = ((last_day_data['Close'] - prev_day_data['Close']) / prev_day_data['Close']) * 100
st.subheader('Resumen de KPIs de los últimos 2 días')
st.write(f'Porcentaje de cambio en el precio: {price_change_percent:.2f}%')


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
else:
    st.write("For an optimal forecasting you need at least 35 days")
    
st.subheader('Raw Data')
st.dataframe(data)
