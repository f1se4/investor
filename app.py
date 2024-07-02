import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mplfinance as mpf


plt.style.use("dark_background")
###########################
#### Funciones Principales
###########################

def get_company_name(ticker):
    # Crear un objeto ticker con yfinance
    ticker_data = yf.Ticker(ticker)
    
    # Obtener el nombre de la empresa
    company_name = ticker_data.info['longName']
    
    return company_name

def get_data(stock, start_time, end_time):
    df = yf.download(stock, start=start_time, end=end_time)
    return df

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, short_window=12, long_window=26):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def plot_candlestick(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    mpf.plot(data,type='candle', style='yahoo',ax=ax)
    return fig

def plot_close_price(data):
    levels = get_levels(data)
    df_levels = pd.DataFrame(levels, columns=['index','close'])
    df_levels.set_index('index', inplace=True)

    fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

    # Plot RSI
    rsi = calculate_rsi(data)
    axs[0].plot(data.index, rsi, color='orange', linewidth=1)
    axs[0].axhline(70, color='red', linestyle='--', linewidth=0.7)
    axs[0].axhline(30, color='green', linestyle='--', linewidth=0.7)
    axs[0].set_ylabel('RSI')
    axs[0].tick_params(axis='x', rotation=45, which='both')
    axs[0].grid(True, color='gray', linestyle='-', linewidth=0.2)

    # Plot MACD
    macd_line, signal_line = calculate_macd(data)
    axs[1].fill_between(data.index, macd_line, where=(macd_line >= 0), color='green', alpha=0.3)
    axs[1].fill_between(data.index, macd_line, where=(macd_line < 0), color='red', alpha=0.3)
    axs[1].plot(data.index, macd_line, label='MACD', color='cyan', linewidth=1)
    axs[1].plot(data.index, signal_line, label='Señal', color='magenta', linewidth=1)
    axs[1].axhline(0, color='gray', linestyle='-', linewidth=0.5)
    axs[1].set_ylabel('MACD')
    axs[1].tick_params(axis='x', rotation=45, which='both')
    axs[1].grid(True, color='gray', linestyle='-', linewidth=0.2)
    axs[1].legend(frameon=False)

    plt.tight_layout()

    return fig

def plot_full(data):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Gráfico del Precio
    ax1.plot(data.index, data.Close, color='dodgerblue', linewidth=1)
    ax1.set_ylabel('Precio USD')
    ax1.tick_params(axis='x', rotation=45, which='both')
    ax1.grid(True, color='gray', linestyle='-', linewidth=0.2)

    plt.tight_layout()

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
    font = {'family': 'sans-serif',
            'color':  'white',
            'weight': 'normal',
            'size': 16,
            }

    font_sub = {'family': 'sans-serif',
            'color':  'white',
            'weight': 'normal',
            'size': 10,
            }


    df_plot = df_vol.copy()
    fig = plt.figure(figsize=(10,6))
    plt.plot(df_plot.index, df_plot.returns, color='dodgerblue', linewidth=0.5)
    plt.plot(df_plot.index, df_plot.volatility, color='darkorange', linewidth=1)
    #mplcyberpunk.add_glow_effects()
    plt.ylabel('% Porcentaje')
    plt.xticks(rotation=45,  ha='right')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.3f}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.grid(True,color='gray', linestyle='-', linewidth=0.2)
    plt.legend(('Retornos Diarios', 'Volatilidad Móvil'), frameon=False)
    return fig


def plot_ma(data):
    # Calcular medias móviles
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()

    # Configurar gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Precio de Cierre', color='dodgerblue', linewidth=1)
    ax.plot(data.index, data['MA_20'], label='MA 20', color='orange', linestyle='--', linewidth=1)
    ax.plot(data.index, data['MA_50'], label='MA 50', color='green', linestyle='--', linewidth=1)
    ax.set_ylabel('Precio USD')
    ax.legend()
    ax.grid(True, color='gray', linestyle='-', linewidth=0.2)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


###########################
#### LAYOUT - Sidebar
###########################

logo = Image.open('assets/logo.jpeg')

# Leer los tickers desde el archivo
with open('tickers.txt', 'r') as file:
    tickers = [line.strip() for line in file.readlines()]

with st.sidebar:
    st.image(logo)
    stock = st.selectbox('Ticker', tickers, index=0)
    start_time = st.date_input(
                    "Fecha de Inicio",
                    datetime.date.today() - datetime.timedelta(days=182))
    end_time = st.date_input(
                    "Fecha Final",
                    datetime.date.today())
    st.header("Help")
    st.image(Image.open('assets/velas.jpg'))

###########################
#### DATA - Funciones sobre inputs
###########################

data = get_data(stock, start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d"))
plot_price = plot_close_price(data)
plot_full_fig = plot_full(data)

df_ret = daily_returns(data)
df_vol = returns_vol(df_ret)
plot_vol = plot_volatility(df_vol)


###########################
#### LAYOUT - Render Final
###########################
company_name = get_company_name(stock)
st.title(f"Análisis {stock}: {company_name}")

st.pyplot(plot_full_fig)

plot_candlestick_fig = plot_candlestick(data)
st.pyplot(plot_candlestick_fig)


st.pyplot(plot_price)
st.markdown("""
    - **RSI (Relative Strength Index):**
      - El RSI es un indicador de momentum que mide la velocidad y el cambio de los movimientos de precios.
      - Se calcula como un valor oscilante entre 0 y 100 y se utiliza comúnmente para identificar condiciones de sobrecompra (por encima de 70) y sobreventa (por debajo de 30).
      - Cuando el RSI está por encima de 70, el activo se considera sobrecomprado, lo que podría indicar una posible corrección a la baja.
      - Por el contrario, cuando el RSI está por debajo de 30, el activo se considera sobrevendido, lo que podría señalar una oportunidad de compra.

    - **MACD (Moving Average Convergence Divergence):**
      - El MACD es un indicador de seguimiento de tendencias que muestra la diferencia entre dos medias móviles exponenciales (EMA) de distintos períodos.
      - Consiste en:
        - **MACD Line:** La diferencia entre el EMA de corto plazo (generalmente 12 períodos) y el EMA de largo plazo (generalmente 26 períodos).
        - **Signal Line:** Una media móvil exponencial de 9 períodos del MACD Line, conocida como línea de señal.
      - Las señales de compra y venta se generan cuando el MACD cruza por encima o por debajo de su línea de señal, respectivamente.
      - Además, las áreas sombreadas en el gráfico indican la tendencia del MACD (verde para alcista y rojo para bajista).
""")

# Renderizar gráfico de medias móviles
st.subheader('Rolling Means')
plot_ma_fig = plot_ma(data)
st.pyplot(plot_ma_fig)

st.subheader('Daily Returns')
st.pyplot(plot_vol)
# Explicación de los retornos diarios y volatilidad
st.markdown("""
    Este gráfico muestra los retornos diarios logarítmicos del precio de cierre del activo y la volatilidad asociada.
    - **Retornos Diarios:** Representan el cambio porcentual en el precio de cierre de un día a otro.
    - **Volatilidad:** Es la desviación estándar de los retornos diarios móviles (últimos 12 días), que indica la variabilidad en los cambios de precio.     - La volatilidad alta puede indicar fluctuaciones significativas en el precio del activo.
""")

st.subheader('Raw Data')
st.dataframe(data)
