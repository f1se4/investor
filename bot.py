import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
# from ib_insync import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from calculations.calculations import get_company_name

# Función para calcular la EMA
def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

# Función para calcular el RSI
def rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df):
    df['EMA_12'] = ema(df['Close'], 12)
    df['EMA_26'] = ema(df['Close'], 26)
    df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD'] = df['MACD_Line'] - df['Signal_Line']
    return df

# Función para obtener el mínimo y máximo de los últimos N días
def rolling_min_max(series, window):
    rolling_min = series.rolling(window=window).min()
    rolling_max = series.rolling(window=window).max()
    return rolling_min, rolling_max

# Función para calcular las bandas de Bollinger
def bollinger_bands(series, window):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def calculate_g_channel(df, window=100):
    df['High'] = df['High'].rolling(window=window).max()
    df['Low'] = df['Low'].rolling(window=window).min()
    df['Mid'] = (df['High'] + df['Low']) / 2
    return df

# def calculate_g_channel(data, length=100):
#     # Rellenar valores nulos usando el método 'bfill'
#     src_filled = data['Close']
#     nz_previous = src_filled.shift(1).fillna(method='bfill')  # Rellena los NaN con el siguiente valor no nulo
#     previous = src_filled.shift(1)
#
#     amax = pd.concat([src_filled,nz_previous], axis=1).max(axis=1)
#     bmin = pd.concat([src_filled,nz_previous], axis=1).min(axis=1)
#
#     a = amax - ((amax - bmin).fillna(method='bfill'))/length
#     b = bmin + ((amax - bmin).fillna(method='bfill'))/length
#
#     # Calcular el promedio
#     data['a'] = a
#     data['b'] = b
#     data['avg'] = (a + b) / 2
#     close = data['Close']
#
#     # Calcular las condiciones de cruce
#     crossup = (b.shift(1) < close.shift(1)) & (b > close)
#     crossdn = (a.shift(1) < close.shift(1)) & (a > close)
#     
#     # Determinar si es una tendencia alcista
#     data['crossup'] = crossup
#     data['crossdn'] = crossdn
#     #
#     # Calcular barssince para crossdn y crossup
#     barssince_crossdn = crossdn.groupby((crossdn != crossdn.shift()).cumsum()).cumcount()
#     barssince_crossup = crossup.groupby((crossup != crossup.shift()).cumsum()).cumcount()
#     
#     # Calcular la condición bullish
#     bullish = barssince_crossdn <= barssince_crossup
#     data['bullish'] = bullish
#
#     return data

# Función para obtener los datos históricos
def get_data(ticker, selected_interval):
    data = yf.download(ticker, period='1d', interval=selected_interval)
    # data = calculate_g_channel(data)
    # print(data)
    data['EMA_50'] = ema(data['Close'], window=50)
    data['EMA_200'] = ema(data['Close'], window=200)
    data['RSI'] = rsi(data['Close'], window=14)
    data['Min_14'], data['Max_14'] = rolling_min_max(data['Close'], window=14)
    data['Bollinger_High'], data['Bollinger_Low'] = bollinger_bands(data['Close'], window=20)
    data = calculate_macd(data)
    data['Volume_Avg'] = data['Volume'].rolling(window=20).mean()
    data['High_Rolling'] = data['High'].rolling(window=14).max()
    data['High_Rolling_Rounded'] = data['High_Rolling'].round(2)
    data['Low_Rolling'] = data['Low'].rolling(window=14).min()
    data['Low_Rolling_Rounded'] = data['Low'].rolling(window=14).min()
    # data['atr'] = atr(data, 14)

    # Calcular las señales de ruptura
    volume_threshold = 1.5
    data['Breakout_Above'] = (data['Close'] > data['High_Rolling']) 
    data['Breakout_Volume'] = (data['Volume'] > volume_threshold * data['Volume_Avg'])
    data['Breakout_Below'] = (data['Close'] < data['Low_Rolling'])

    return data

# Average True Range (ATR)
def atr(df, window):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window).mean()

# Función para generar señales de trading
def generate_signals(data, show_g_channel, show_simple_trade):
    # A - Swing Trading
    #      * Identify Oportunities based in trends and correction cycles
    # B - Breakouts 
    #      * levels and confirmations
    if show_simple_trade:
        data['Buy_Signal'] = np.where((data['Close'] < data['EMA_200']) & #A
                                      # (data['RSI'] < 30) & #A
                                      (data['Close'] > data['Bollinger_High']) &
                                      # (data['Close'] <= data['Min_14']) &
                                      (data['MACD'] > 0 ) & #A
                                      # (data['Breakout_Above']) &
                                      (data['Breakout_Volume']), 1, 0) #B

        data['Sell_Signal'] = np.where((data['Close'] > data['EMA_200']) &
                                       # (data['RSI'] > 70) &
                                       (data['Close'] < data['Bollinger_Low']) &
                                       # (data['Close'] >= data['Max_14']) &
                                       (data['MACD'] < 0 ) &
                                       # (data['Breakout_Below']) & #B
                                      (data['Breakout_Volume']), 1, 0) #B
    if show_g_channel:
        pass
        # buy_signals = np.where((data['bullish'] & ~np.roll(data['bullish'], 1)), data['avg'], np.nan)
        # sell_signals = np.where((~data['bullish'] & np.roll(data['bullish'], 1)), data['avg'], np.nan)
        # data['Buy_Signal_GC'] = buy_signals
        # data['Sell_Signal_GC'] = sell_signals

    data['Buy'] = data.get('Buy_Signal', 0) + data.get('Buy_Signal_GC', 0)
    data['Sell'] = data.get('Sell_Signal', 0) + data.get('Sell_Signal_GC', 0)

    return data

# Función para determinar la acción a tomar
def determine_action(data, position):
    if position == 'None':
        if data.iloc[-1]['Buy'] >= 1:
            return 'Buy', data.index[-1]
        else:
            return 'Hold', None
    elif position == 'Long':
        if data.iloc[-1]['Sell'] >= 1:
            return 'Sell', data.index[-1]
        else:
            return 'Hold', None

# Función para realizar operaciones con Interactive Brokers
# def place_order(ticker, action, quantity, simulate=True):
#     ib = IB()
#     if simulate:
#         ib.connect('127.0.0.1', 7497, clientId=1)  # Simulated trading
#     else:
#         ib.connect('127.0.0.1', 7496, clientId=1)  # Live trading
#     
#     contract = Stock(ticker, 'SMART', 'USD')
#     ib.qualifyContracts(contract)
#     
#     if action == 'Comprar':
#         order = MarketOrder('BUY', quantity)
#     elif action == 'Vender':
#         order = MarketOrder('SELL', quantity)
#     else:
#         return "No action taken"
#     
#     trade = ib.placeOrder(contract, order)
#     ib.sleep(1)
#     ib.disconnect()
#     return trade
#
# Función para actualizar la cartera
def update_portfolio(ticker, action, quantity, price):
    portfolio_file = 'portfolio.csv'
    
    if not os.path.exists(portfolio_file):
        df = pd.DataFrame(columns=['Ticker', 'Action', 'Quantity', 'Price', 'Date'])
    else:
        df = pd.read_csv(portfolio_file)
    
    new_trade = {
        'Ticker': ticker,
        'Action': action,
        'Quantity': quantity,
        'Price': price,
        'Date': pd.Timestamp.now()
    }
    df = df.append(new_trade, ignore_index=True)
    df.to_csv(portfolio_file, index=False)
    return df

# Función para mostrar la cartera
def show_portfolio():
    portfolio_file = 'portfolio.csv'
    
    if os.path.exists(portfolio_file):
        df = pd.read_csv(portfolio_file)
        return df
    else:
        return pd.DataFrame(columns=['Ticker', 'Action', 'Quantity', 'Price', 'Date'])

# Función para graficar datos con Plotly
def plot_data(data, ticker, show_g_channel, show_simple_trade):
    company_name = get_company_name(ticker)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        row_heights=[0.8, 0.2, 0.2],
                        vertical_spacing=0.05)

    data = data.tail(60).copy()

    # Añadir gráfico de velas (candlestick)
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'), row=1, col=1)

    # fig.add_trace(go.Scatter(x=data.index, y=data['EMA_200'], mode='lines', name='EMA 200',
    #                          line=dict(color='rgba(255,255,204, 0.8)')))

    fig.add_trace(go.Bar(x=data.index, y=data.MACD, 
                         marker_color=np.where(data.MACD >= 0, 'green', 'darkgray'), 
                         opacity=0.6), row=3, col=1)

    # Añadir gráfico de volumen al segundo subplot
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='rgba(31, 119, 180, 0.3)'),
                  row=2, col=1)
    # Añadir gráfico de volumen al segundo subplot
    fig.add_trace(go.Scatter(x=data.index, y=data['Volume_Avg'], name='Volume', 
                             marker_color='rgba(131, 119, 180, 0.4)'),
                  row=2, col=1)

    if show_simple_trade:
        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_High'], mode='lines', name='Bollinger High',
                                 line=dict(color='rgba(214, 39, 40, 0.3)')))

        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Low'], mode='lines', name='Bollinger Low',
                                 line=dict(color='rgba(255, 152, 150, 0.3)')))  # Color similar a rgba(214, 39, 40, 0.3)

        # Añadir área sombreada entre Bollinger High y Bollinger Low
        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_High'], fill='tonexty',
                                 fillcolor='rgba(214, 39, 40, 0.1)', line=dict(color='rgba(214, 39, 40, 0.3)'),
                                 mode='lines', name='Bollinger Bands'))
        
        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Low'], fill='tonexty',
                                 fillcolor='rgba(255, 152, 150, 0.1)', line=dict(color='rgba(255, 152, 150, 0.3)'),
                                 mode='lines', name='Bollinger Bands'))

        buy_signals = data[data['Buy_Signal'] == 1]
        sell_signals = data[data['Sell_Signal'] == 1]

        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers+text', name='Buy Signal',
                                 marker=dict(color='magenta', size=7, symbol="cross"), 
                                 text=buy_signals.index.strftime('%Y-%m-%d %H:%M'),
                                 textposition="bottom left", textfont=dict(color='magenta')))

        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers+text', name='Sell Signal',
                                 marker=dict(color='orange', size=7, symbol="x"), 
                                 text=sell_signals.index.strftime('%Y-%m-%d %H:%M'),
                                 textposition="top left",textfont=dict(color='orange')))

    if show_g_channel:
        pass
        # Agregar líneas de promedio y precios de cierre
        # fig.add_trace(go.Scatter(x=data.index, y=data['avg'], mode='lines', name='Average', line=dict(color='green', width=1)))
        # fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close price', line=dict(color='blue', width=1)))

        #Agregar señales de compra/venta
        # fig.add_trace(go.Scatter(x=data.index, y=data['Buy_Signal_GC'], mode='markers', name='Buy', marker=dict(color='lime', size=10, symbol='triangle-up')))
        # fig.add_trace(go.Scatter(x=data.index, y=data['Sell_Signal_GC'], mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')))
        # buy_signals = data[data['Buy_Signal_GC'] == 1]
        # sell_signals = data[data['Sell_Signal_GC'] == -1]
        # 
        # fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
        #                          mode='markers', marker=dict(symbol='triangle-up', color='magenta', size=10), name='Buy Signal'))
        # fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
        #                          mode='markers', marker=dict(symbol='triangle-down', color='orange', size=10), name='Sell Signal'))

        # Add shaded areas for the G-Channel
        # fig.add_trace(go.Scatter(x=data.index, y=data['High'], line=dict(color='green', width=1), name='High Channel'))
        # fig.add_trace(go.Scatter(x=data.index, y=data['Low'], line=dict(color='red', width=1), name='Low Channel', fill='tonexty', fillcolor='rgba(0, 100, 80, 0.2)'))

    fig.update_layout(title=f'{ticker} - {company_name}', 
                      xaxis_title='', yaxis_title='', 
                      xaxis_rangeslider_visible=False,
                      showlegend=False)

    return fig

# Función principal
def bot_main(selected_interval='5m', show_g_channel=True, show_simple_trade=True, tickers="BTC-EUR"):
    st.title("TradeBot")

    tickers = [ticker.strip() for ticker in tickers.split(',')]
    # simulate = st.checkbox("Simular operaciones", value=True)
    
    if tickers:
        actions = []
        portfolio = show_portfolio()
        current_positions = {ticker: 'None' for ticker in tickers}

        if not portfolio.empty:
            for ticker in tickers:
                if (portfolio['Ticker'] == ticker).any():
                    last_action = portfolio[portfolio['Ticker'] == ticker].iloc[-1]['Action']
                    current_positions[ticker] = 'Long' if last_action == 'Comprar' else 'None'
        
        for ticker in tickers:
            data = get_data(ticker, selected_interval)
            try:
                data = generate_signals(data, show_g_channel, show_simple_trade)
                action, signal_date = determine_action(data, current_positions[ticker])
                actions.append({'Ticker': ticker, 'Acción': action, 'Fecha de Señal': signal_date})
                
                # st.write(data.tail(10))
                st.plotly_chart(plot_data(data, ticker, show_g_channel, show_simple_trade))
                with st.expander(f'data - {ticker}'):
                    st.dataframe(data)
                # if action != 'No hacer nada':
                #     # trade = place_order(ticker, action, quantity=10, simulate=simulate)  # Cantidad fija de 10 para ejemplo
                #     st.write(f"Orden ejecutada para {ticker}: trade")
                #     
                #     # Actualizar la cartera
                #     last_price = data.iloc[-1]['Close']
                #     portfolio = update_portfolio(ticker, action, 10, last_price)
            except:
                st.write(f"Error getting {ticker}")


        st.subheader("Acciones a Tomar")
        st.dataframe(pd.DataFrame(actions))
        
        st.subheader("Cartera Actual")
        portfolio_df = show_portfolio()
        st.write(portfolio_df)
