import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

#########################################################################################
#### Funciones Cálculos
#########################################################################################
# Función para formatear valores con color
def format_value(value):
    color = "green" if value >= 0 else "red"
    return f"<span style='color:{color}'>{value:.2f}</span>"

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
    prices = data['Close'].dropna()
    model = ARIMA(prices, order=(3, 1, 3))
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
