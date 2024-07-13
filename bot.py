import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
# from ib_insync import *
import os

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
    data = yf.download(ticker, start="2020-01-01")
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
        return 'Comprar'
    elif data.iloc[-1]['Sell_Signal'] == 1:
        return 'Vender'
    else:
        return 'No hacer nada'

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

# Función principal
def bot_main():
    st.title("Bot de Trading para Múltiples Tickers con Interactive Brokers")
    
    tickers = st.text_area("Introduce los símbolos de los tickers separados por comas", "AAPL, MSFT, TSLA")
    tickers = [ticker.strip() for ticker in tickers.split(',')]
    simulate = st.checkbox("Simular operaciones", value=True)
    
    if tickers:
        actions = []
        
        for ticker in tickers:
            try:
                data = get_data(ticker)
                data = generate_signals(data)
                action = determine_action(data)
                actions.append({'Ticker': ticker, 'Acción': action})
                
                st.subheader(f"Ticker: {ticker}")
                # st.write(data.tail(10))
                buy_signals = data[data['Buy_Signal'] == 1]
                chart = st.line_chart(data[['Close', 'EMA_50', 'EMA_200', 'Bollinger_High', 'Bollinger_Low']])
                for idx, row in buy_signals.iterrows():
                    st.text(f"Compra el {idx.strftime('%Y-%m-%d')} ({row['Close']})")
                    chart.add_line(series=data.loc[[idx]], color='magenta', name=f"Compra {idx.strftime('%Y-%m-%d')}")
            
                sell_signals = data[data['Sell_Signal'] == 1]
                for idx, row in sell_signals.iterrows():
                    st.text(f"Venta el {idx.strftime('%Y-%m-%d')} ({row['Close']})")
                    chart.add_line(series=data.loc[[idx]], color='orange', name=f"Venta {idx.strftime('%Y-%m-%d')}")
                
                # st.write("Señales de Compra")
                # st.write(data[data['Buy_Signal'] == 1][['Close', 'EMA_50', 'EMA_200', 'RSI', 'Volume']])
                # 
                # st.write("Señales de Venta")
                # st.write(data[data['Sell_Signal'] == 1][['Close', 'EMA_50', 'EMA_200', 'RSI', 'Volume']])
                # 
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
        st.write(actions_df)
        
        st.subheader("Cartera Actual")
        portfolio_df = show_portfolio()
        st.write(portfolio_df)
