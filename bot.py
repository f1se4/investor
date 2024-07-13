import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
# from ib_insync import *
import plotly.graph_objects as go
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

# Función para calcular las bandas de Bollinger
def bollinger_bands(series, window):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

# Función para obtener los datos históricos
def get_data(ticker):
    data = yf.download(ticker, period='10y', interval='1d')
    data['EMA_50'] = ema(data['Close'], window=50)
    data['EMA_200'] = ema(data['Close'], window=200)
    data['RSI'] = rsi(data['Close'], window=14)
    data['Bollinger_High'], data['Bollinger_Low'] = bollinger_bands(data['Close'], window=20)
    data['Volume_Avg'] = data['Volume'].rolling(window=20).mean()
    return data

# Función para generar señales de trading
def generate_signals(data):
    data['Buy_Signal'] = np.where((data['EMA_50'] > data['EMA_200']) &
                                  (data['RSI'] < 30) &
                                  (data['Close'] <= data['Bollinger_Low']) &
                                  (data['Volume'] > 1.5 * data['Volume_Avg']), 1, 0)
    
    data['Sell_Signal'] = np.where((data['EMA_50'] < data['EMA_200']) &
                                   (data['RSI'] > 70) &
                                   (data['Close'] >= data['Bollinger_High']) &
                                   (data['Volume'] > 1.5 * data['Volume_Avg']), 1, 0)
    return data
# Función para determinar la acción a tomar
def determine_action(data):
    if data.iloc[-1]['Buy_Signal'] == 1:
        return 'Comprar', data.index[-1]
    elif data.iloc[-1]['Sell_Signal'] == 1:
        return 'Vender', data.index[-1]
    else:
        return 'No hacer nada', None

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
def plot_data(data, ticker):
    company_name = get_company_name(ticker)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close',
                             line=dict(color='rgba(31, 119, 180, 0.8)')))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], mode='lines', name='EMA 50',
                             line=dict(color='rgba(255, 127, 14, 0.3)')))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_200'], mode='lines', name='EMA 200',
                             line=dict(color='rgba(44, 160, 44, 0.3)')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_High'], mode='lines', name='Bollinger High',
                             line=dict(color='rgba(214, 39, 40, 0.3)')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Low'], mode='lines', name='Bollinger Low',
                             line=dict(color='rgba(148, 103, 189, 0.3)')))    
    
    buy_signals = data[data['Buy_Signal'] == 1]
    sell_signals = data[data['Sell_Signal'] == 1]
    
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers+text', name='Buy Signal',
                             marker=dict(color='magenta', size=10, symbol="cross"), text=buy_signals.index.strftime('%Y-%m-%d'),
                             textposition="bottom center", textfont=dict(color='magenta')))
    
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers+text', name='Sell Signal',
                             marker=dict(color='orange', size=10, symbol="x"), text=sell_signals.index.strftime('%Y-%m-%d'),
                             textposition="top center",textfont=dict(color='orange')))
    
    fig.update_layout(title=f'{ticker} - {company_name}', xaxis_title='Date', yaxis_title='Price')
    
    return fig

# Función principal
def bot_main():
    st.title("TradeBot")

    #acciones_evaluar = '''AAPL, MSFT, AMZN, GOOGL, TSLA, NVDA, META, JPM, V, NFLX, BABA, AMD, META, SQ, BTC-EUR, ETH-EUR, SPY, QQQ, GLD, SLV, UBER, LYFT, CRM, BA, GE, IBM, SNAP, GM, SBUX, MCD, KO, PFE, MRNA, XOM, CVX, T, VZ, TSM, INTC, SHOP, ZM, DOCU, NIO'''
    acciones_evaluar = "AAPL, MSFT, AMZN, GOOGL, TSLA, NVDA, META, JPM"
    
    tickers = st.text_area("Introduce los símbolos de los tickers separados por comas", acciones_evaluar)
    tickers = [ticker.strip() for ticker in tickers.split(',')]
    simulate = st.checkbox("Simular operaciones", value=True)
    
    if tickers:
        actions = []
        
        for ticker in tickers:
            data = get_data(ticker)
            data = generate_signals(data)
            action, signal_date = determine_action(data)
            try:
                actions.append({'Ticker': ticker, 'Acción': action, 'Fecha de Señal': signal_date})
                
                # st.write(data.tail(10))
                st.plotly_chart(plot_data(data, ticker))
                # if action != 'No hacer nada':
                #     # trade = place_order(ticker, action, quantity=10, simulate=simulate)  # Cantidad fija de 10 para ejemplo
                #     st.write(f"Orden ejecutada para {ticker}: trade")
                #     
                #     # Actualizar la cartera
                #     last_price = data.iloc[-1]['Close']
                #     portfolio = update_portfolio(ticker, action, 10, last_price)
            except:
                pass
        
        st.subheader("Acciones a Tomar")
        actions_df = pd.DataFrame(actions)
        st.dataframe(actions_df)
        
        st.subheader("Cartera Actual")
        portfolio_df = show_portfolio()
        st.write(portfolio_df)