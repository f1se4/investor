import yfinance as yf
import pandas as pd
import numpy as np
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
    gain = (delta.where(delta > 0, 0)).rolling(window=window,  min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df):
    df['EMA_12'] = ema(df['Close'], 12)
    df['EMA_26'] = ema(df['Close'], 26)
    df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD'] = df['MACD_Line'] - df['Signal_Line']
    return df

# Función para calcular las bandas de Bollinger
def bollinger_bands(series, window):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

# Función para obtener los datos históricos
def get_data(ticker, selected_interval, select_period):
    data = yf.download(ticker, period=select_period, interval=selected_interval)
    try:
        data.index = data.index.tz_convert('CET')
    except:
        pass
    data['EMA_50'] = ema(data['Close'], window=50)
    data['EMA_200'] = ema(data['Close'], window=200)
    data['EMA_80'] = ema(data['Close'], window=80)
    data['EMA_280'] = ema(data['Close'], window=280)
    data['RSI'] = rsi(data['Close'], window=14)
    data['Bollinger_High'], data['Bollinger_Low'] = bollinger_bands(data['Close'], window=20)
    data = calculate_macd(data)
    data['Volume_Avg'] = data['Volume'].rolling(window=20).mean()
    data['High_Rolling'] = data['High'].rolling(window=14).max()
    data['Low_Rolling'] = data['Low'].rolling(window=14).min()

    return data

# Función para generar señales de trading
def generate_signals(data, show_MACD, show_simple_trade, show_MM):
    data['Buy'] = 0
    data['Sell'] = 0

    if show_MM: # Diario
        data['Buy'] = np.where((data['EMA_50'] > data['EMA_200']) &
                                      (data['EMA_50'].shift(1) <= data['EMA_200'].shift(1)) &
                                      (data['RSI'] < 50) 
                                      , 1, 0)
                                      # (data['Breakout_Volume']), 1, 0)
        data['Sell'] = np.where((data['EMA_50'] < data['EMA_200']) &
                                      (data['EMA_50'].shift(1) >= data['EMA_200'].shift(1)) &
                                      (data['RSI'] < 50) 
                                      , 1, 0)
                                      # (data['Breakout_Volume']), 1, 0)

    if show_MACD: #Intradia
        data['Buy'] = np.where((data['Close'] < data['Bollinger_Lower']) &
                                          (data['MACD'] > data['Signal_Line'] ), 1,0)

        data['Sell'] = np.where((data['Close'] > data['Bollinger_High']) &
                                          (data['MACD'] > data['Signal_Line'] ), 1,0)
    return data

# Función para determinar la acción a tomar
def determine_action(data, position):
    if position == 'None':
        if data.iloc[-1]['Buy'] >= 1:
            return 'Buy', data.index[-1], data.index[-1]['Close']
        else:
            return 'Hold', None, None
    elif position == 'Long':
        if data.iloc[-1]['Sell'] >= 1:
            return 'Sell', data.index[-1], data.index[-1]['Close']
        else:
            return 'Hold', None, None

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

def calculate_poc_val_vah(data):
    # Crear una DataFrame con precios y volúmenes
    price_volume_df = data[['Close', 'Volume']].copy()
    
    # Ordenar por precios y volúmenes
    price_volume_df = price_volume_df.sort_values(by='Close')
    
    # Calcular el volumen total
    total_volume = price_volume_df['Volume'].sum()
    
    # Calcular el volumen objetivo (70% del volumen total)
    target_volume = total_volume * 0.70
    
    # Inicializar variables para encontrar VAL y VAH
    cumulative_volume = 0
    val = None
    vah = None
    
    # Calcular VAL y VAH
    for price, volume in price_volume_df.itertuples(index=False):
        cumulative_volume += volume
        if cumulative_volume <= target_volume:
            val = price
        else:
            vah = price
            break
    
    # Calcular el POC (precio con mayor volumen)
    poc = data['Volume'].idxmax()
    poc_price = data.loc[poc, 'Close']
    
    return poc_price, val, vah

# Función para graficar datos con Plotly
def plot_data(data, ticker, show_g_channel, show_simple_trade, show_MM):
    format = '%Y-%m-%d %H:%M'
    company_name = get_company_name(ticker)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        row_heights=[0.8, 0.066, 0.066, 0.066],
                        vertical_spacing=0.05)

    # Añadir gráfico de velas (candlestick)
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'), row=1, col=1)
    # Calcular POC, VAL y VAH
    poc_price, val, vah = calculate_poc_val_vah(data)
    
    # Identificar otros máximos relativos en el volumen
    volume_peaks = data['Volume'][(data['Volume'].shift(1) < data['Volume']) & (data['Volume'].shift(-1) < data['Volume'])]

    # Marcar el POC, VAL y VAH en el gráfico de precios
    fig.add_trace(go.Scatter(
        x=[data.index[0], data.index[-1]],
        y=[poc_price, poc_price],
        mode='lines',
        name='POC',
        line=dict(color='rgba(68,102,119,0.8)', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=[data.index[0], data.index[-1]],
        y=[val, val],
        mode='lines',
        name='VAL',
        line=dict(color='rgba(107,107,107,0.5)', dash='dash'))
    )
    
    fig.add_trace(go.Scatter(
        x=[data.index[0], data.index[-1]],
        y=[vah, vah],
        mode='lines',
        name='VAH',
        line=dict(color='rgba(107,107,107,0.5)', dash='dash')
    ))
    
    # Marcar otros máximos relativos
    peak_lines = []
    for peak_date, peak_volume in volume_peaks.items():
        peak_price = data.loc[peak_date, 'Close']
        peak_lines.append(fig.add_trace(go.Scatter(
            y=[peak_price, peak_price],
            x=[data.index[0], data.index[-1]],
            mode='lines',
            name=f'Peak {peak_date.date()}',
            line=dict(color='rgba(93,93,93,0.4)', dash='dot')
        )))

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
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', 
                             marker_color='rgba(131, 119, 180, 0.4)'),
                  row=4, col=1)

    if show_simple_trade:
        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_High'], mode='lines', name='Bollinger High',
                                 line=dict(color='rgba(248, 237, 98, 0.3)')))

        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Low'], mode='lines', name='Bollinger Low',
                                 line=dict(color='rgba(233,215,0, 0.3)')))  # Color similar a rgba(214, 39, 40, 0.3)

        # Añadir área sombreada entre Bollinger High y Bollinger Low
        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_High'], fill='tonexty',
                                 fillcolor='rgba(248,237,98, 0.1)', line=dict(color='rgba(248, 237, 98, 0.3)'),
                                 mode='lines', name='Bollinger Bands'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Low'], fill='tonexty',
                                 fillcolor='rgba(248,237,98, 0.1)', line=dict(color='rgba(214,39,40, 0.3)'),
                                 mode='lines', name='Bollinger Bands'))


    if show_g_channel:
        pass

    if show_MM:
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_200'], mode='lines', name='EMA 200',
                              line=dict(color='rgba(153,204,255, 0.8)')))
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], mode='lines', name='EMA_50', line=dict(color='rgba(85,136,255,0.8)', width=1)))

        format = '%d-%m-%Y'

    buy_signals = data[data['Buy'] == 1]
    sell_signals = data[data['Sell'] == 1]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers+text', name='Buy Signal',
                                 marker=dict(color='#65fe08', size=15, symbol="arrow-up"), 
                                 text=buy_signals.index.strftime(format),
                                 textposition="bottom left", textfont=dict(color='#65fe08')))

    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers+text', name='Sell Signal',
                                 marker=dict(color='#F93822', size=12, symbol="arrow-down"), 
                                 text=sell_signals.index.strftime(format),
                                 textposition="top right",textfont=dict(color='#F93822')))

    fig.update_layout(title=f'{ticker} - {company_name}', 
                      xaxis_title='', yaxis_title='', 
                      xaxis_rangeslider_visible=False,
                      height=700,
                      showlegend=False)

    return fig

def f_backtesting(data):
    # DataFrame para almacenar los resultados del backtesting
    results = []
    old_price = 0
    counter = 0
    
    # Recorrer el DataFrame `data` para identificar las operaciones
    for index, row in data.iterrows():
        if row['Buy'] == 1.0 or row['Sell'] == 1.0:
            print(row['Buy'],row['Sell'])
            buy_price = 0
            sell_price = 0
            trade_return = 0

            if row['Buy'] == 1:
                buy_price = row['Close']
                signal_date = index
                action = 'Buy'
                if counter == 0:
                    counter =+ 1
                    trade_return = 0
                else:
                    # Registrar la compra
                    sell_price=old_price
                    try:
                        trade_return = (sell_price - buy_price) / buy_price
                    except:
                        trade_return = 0
                old_price = buy_price
            elif row['Sell'] == 1:
                if counter == 0:
                    continue
                # Registrar la venta
                buy_price=old_price
                sell_price = row['Close']
                signal_date = index
                action = 'Sell'
                old_price=sell_price
                # Calcular el rendimiento de la operación
                try:
                    trade_return = (sell_price - buy_price) / buy_price
                except:
                    trade_return = 0
            # Añadir la operación al DataFrame de resultados
            results.append({
                    'Date': signal_date,
                    'Action' : action,
                    'Buy_Price': buy_price,
                    'Sell_Price': sell_price,
                    'Return': trade_return
                })
    return pd.DataFrame(results)

# Función para aplicar formato condicional y otros estilos
def style_dataframe(df):
    df['Return_Percent'] = df['Return'] * 100  # Convertir a porcentaje para barras de progreso
    df['Return'] = df['Return'].apply(lambda x: "{:.2%} 💰️".format(x))  # Formatear como porcentaje

    styled_df = df.style.applymap(
        lambda x: 'color: red;' if isinstance(x, str) and '-' in x else 'color: green;' if isinstance(x, str) else '',
        subset=['Return'])

    return styled_df
