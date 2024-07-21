import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from calculations.calculations import get_company_name
from datetime import timedelta

def f_parabolic_SAR(df, af=0.03, max_af=0.3):
    """
    Calcula el Parabolic SAR para un dataframe de pandas con columnas 'High' y 'Low'.

    Parameters:
    df (pandas.DataFrame): DataFrame con los datos de precios.
    af (float): Factor de aceleración inicial.
    max_af (float): Factor de aceleración máximo.

    Returns:
    pandas.DataFrame: DataFrame con una columna adicional 'SAR' con los valores del Parabolic SAR.
    """
    high = df['High']
    low = df['Low']
    
    sar = df['Close'].copy()
    uptrend = True
    af = af
    ep = high.iloc[0]
    
    for i in range(1, len(df)):
        if uptrend:
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            if low.iloc[i] < sar.iloc[i]:
                uptrend = False
                sar.iloc[i] = ep
                af = 0.02
                ep = low.iloc[i]
        else:
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            if high.iloc[i] > sar.iloc[i]:
                uptrend = True
                sar.iloc[i] = ep
                af = 0.02
                ep = high.iloc[i]
        
        if uptrend:
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + 0.02, max_af)
        else:
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + 0.02, max_af)
    
    df['SAR'] = sar
    return df

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

# Identificar divergencias
def identify_rsi_divergences(df):
    divergences = []
    for i in range(1, len(df) - 1):
        if df['Close'][i] < df['Close'][i-1] and df['Close'][i] < df['Close'][i+1]:
            if df['RSI'][i] > df['RSI'][i-1] and df['RSI'][i] > df['RSI'][i+1]:
                divergences.append((df.index[i], df['Close'][i], df['RSI'][i], 'Bullish'))
        if df['Close'][i] > df['Close'][i-1] and df['Close'][i] > df['Close'][i+1]:
            if df['RSI'][i] < df['RSI'][i-1] and df['RSI'][i] < df['RSI'][i+1]:
                divergences.append((df.index[i], df['Close'][i], df['RSI'][i], 'Bearish'))
    return divergences

def identify_rsi_divergences(df):
    divergences = []
    for i in range(1, len(df) - 1):
        # Divergencia Alcista: Precio haciendo un mínimo más bajo y RSI haciendo un mínimo más alto
        if df['Close'][i] < df['Close'][i-1] and df['Close'][i] < df['Close'][i+1]:
            if df['RSI'][i] > df['RSI'][i-1] and df['RSI'][i] > df['RSI'][i+1]:
                divergences.append((df.index[i], df['RSI'][i], 'Bullish'))
        # Divergencia Bajista: Precio haciendo un máximo más alto y RSI haciendo un máximo más bajo
        if df['Close'][i] > df['Close'][i-1] and df['Close'][i] > df['Close'][i+1]:
            if df['RSI'][i] < df['RSI'][i-1] and df['RSI'][i] < df['RSI'][i+1]:
                divergences.append((df.index[i], df['RSI'][i], 'Bearish'))
    return divergences

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
def get_data(ticker, selected_interval, select_period, smai=200, smaii=100):
    data = yf.download(ticker, period=select_period, interval=selected_interval)
    try:
        data.index = data.index.tz_convert('CET')
    except:
        pass

    data["Datetime"] = data.index

    # # Verificamos el intervalo seleccionado
    # if 'd' in selected_interval:  # Si el intervalo es diario o superior
    #     data = data.asfreq('D')
    #     data.fillna(method='ffill', inplace=True)
    #     data = data[data.index.dayofweek < 5]
    # else:  # Si el intervalo es intradía
    #     data = data.asfreq(selected_interval)
    #     data.fillna(method='ffill', inplace=True)

    data['SMAI'] = data['Close'].rolling(window=smai).mean()
    data['SMAII'] = data['Close'].rolling(window=smaii).mean()

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
    data = f_parabolic_SAR(data)

    return data

# Función para generar señales de trading
# def generate_signals(data, show_MACD, show_simple_trade, show_MM):
#     data['Buy'] = 0
#     data['Sell'] = 0
#
#     if show_MM: # Diario
#         data['Buy'] = np.where((data['EMA_50'] > data['EMA_200']) &
#                                       (data['EMA_50'].shift(1) <= data['EMA_200'].shift(1)) &
#                                       (data['RSI'] < 50) 
#                                       , 1, 0)
#                                       # (data['Breakout_Volume']), 1, 0)
#         data['Sell'] = np.where((data['EMA_50'] < data['EMA_200']) &
#                                       (data['EMA_50'].shift(1) >= data['EMA_200'].shift(1)) &
#                                       (data['RSI'] < 50) 
#                                       , 1, 0)
#                                       # (data['Breakout_Volume']), 1, 0)
#
#     if show_MACD: #Intradia
#         data['Buy'] = np.where((data['Close'] < data['Bollinger_Lower']) &
#                                           (data['MACD'] > data['Signal_Line'] ), 1,0)
#
#         data['Sell'] = np.where((data['Close'] > data['Bollinger_High']) &
#                                           (data['MACD'] > data['Signal_Line'] ), 1,0)
#     return data

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
    
    # Calcular el POC (precio con mayor volumen)
    poc_index = data['Volume'].idxmax()
    poc_price = data.loc[poc_index, 'Close']
    
    # Ordenar por precios
    price_volume_df = price_volume_df.sort_values(by='Close')
    
    # Calcular el volumen total
    total_volume = price_volume_df['Volume'].sum()
    
    # Calcular el volumen objetivo (70% del volumen total)
    target_volume = total_volume * 0.70
    
    # Inicializar variables para encontrar VAL y VAH
    cumulative_volume = 0
    val = None
    vah = None
    middle_price = poc_price

    # Encuentra VAL (Value Area Low) y VAH (Value Area High) alrededor del POC
    lower_half_volume = price_volume_df[price_volume_df['Close'] <= middle_price]
    upper_half_volume = price_volume_df[price_volume_df['Close'] > middle_price]

    lower_cumulative_volume = lower_half_volume['Volume'].sum()
    upper_cumulative_volume = upper_half_volume['Volume'].sum()

    if lower_cumulative_volume > target_volume / 2:
        # Encuentra el VAL en la mitad inferior
        lower_half_volume = lower_half_volume.sort_values(by='Close', ascending=False)
        cumulative_volume = 0
        for price, volume in lower_half_volume.itertuples(index=False):
            cumulative_volume += volume
            if cumulative_volume >= target_volume / 2:
                val = price
                break
    else:
        val = lower_half_volume['Close'].min()

    if upper_cumulative_volume > target_volume / 2:
        # Encuentra el VAH en la mitad superior
        upper_half_volume = upper_half_volume.sort_values(by='Close')
        cumulative_volume = 0
        for price, volume in upper_half_volume.itertuples(index=False):
            cumulative_volume += volume
            if cumulative_volume >= target_volume / 2:
                vah = price
                break
    else:
        vah = upper_half_volume['Close'].max()
    
    return poc_price, val, vah

# Función para graficar datos con Plotly
# def plot_data(data, ticker, show_g_channel, show_simple_trade, show_MM, show_MMI, show_par=True):
#     format = '%Y-%m-%d %H:%M'
#     company_name = get_company_name(ticker)
#     fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
#                         row_heights=[0.7, 0.1, 0.1, 0.1],
#                         vertical_spacing=0.05)
#
#     # Añadir gráfico de velas (candlestick)
#     fig.add_trace(go.Candlestick(x=data.index,
#                                  open=data['Open'],
#                                  high=data['High'],
#                                  low=data['Low'],
#                                  close=data['Close'],
#                                  name='Candlestick'), row=1, col=1)
#
#     if show_g_channel: #En verdad es volumen
#         # Calcular POC, VAL y VAH
#         poc_price, val, vah = calculate_poc_val_vah(data)
#         
#         # Identificar otros máximos relativos en el volumen
#         volume_peaks = data['Volume'][(data['Volume'].shift(2) < data['Volume']) & (data['Volume'].shift(-2) < data['Volume'])]
#
#         # Marcar el POC, VAL y VAH en el gráfico de precios
#         fig.add_trace(go.Scatter(
#             x=[data.index[0], data.index[-1]],
#             y=[poc_price, poc_price],
#             mode='lines',
#             name='POC',
#             line=dict(color='rgba(68,102,119,0.8)', dash='dash')
#         ))
#         
#         fig.add_trace(go.Scatter(
#             x=[data.index[0], data.index[-1]],
#             y=[val, val],
#             mode='lines',
#             name='VAL',
#             line=dict(color='rgba(107,107,107,0.5)', dash='dash'))
#         )
#         
#         fig.add_trace(go.Scatter(
#             x=[data.index[0], data.index[-1]],
#             y=[vah, vah],
#             mode='lines',
#             name='VAH',
#             line=dict(color='rgba(107,107,107,0.5)', dash='dash')
#         ))
#         # Marcar otros máximos relativos
#         peak_lines = []
#         for peak_date, peak_volume in volume_peaks.items():
#             peak_price = data.loc[peak_date, 'Close']
#             peak_lines.append(fig.add_trace(go.Scatter(
#                 y=[peak_price, peak_price],
#                 x=[data.index[0], data.index[-1]],
#                 mode='lines',
#                 name=f'Peak {peak_date.date()}',
#                 line=dict(color='rgba(93,93,93,0.1)', dash='dot')
#             )))
#
#     if show_par:
#         fig.add_trace(go.Scatter(x=data.index, y=data['SAR'],
#                              mode='markers',
#                              marker=dict(color='rgba(125,132,113,0.4)', size=5),
#                              name='Parabolic SAR'))
#     
#
#     fig.add_trace(go.Bar(x=data.index, y=data.MACD, 
#                          marker_color=np.where(data.MACD >= 0, 'green', 'darkgray'), 
#                          opacity=0.6), row=3, col=1)
#
#     # Añadir gráfico de volumen al segundo subplot
#     fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='rgba(31, 119, 180, 0.3)'),
#                   row=2, col=1)
#     # Añadir gráfico de volumen al segundo subplot
#     fig.add_trace(go.Scatter(x=data.index, y=data['Volume_Avg'], name='Volume', 
#                              marker_color='rgba(131, 119, 180, 0.4)'),
#                   row=2, col=1)
#     fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', 
#                              marker_color='rgba(131, 119, 180, 0.6)'),
#                   row=4, col=1)
#     fig.add_trace(go.Scatter(
#         mode='lines',
#                 y=[30, 30],
#                 x=[data.index[0], data.index[-1]],
#                              marker_color='rgba(239,169,74,0.3)'),
#                   row=4, col=1)
#     fig.add_trace(go.Scatter(
#                 mode='lines',
#                 y=[70, 70],
#                 x=[data.index[0], data.index[-1]],
#                 marker_color='rgba(239,169,74,0.3)'),
#                 row=4, col=1)
#     # Diverengcias RSI
#     divergences = identify_rsi_divergences(data)
#     #Agregar divergencias al gráfico# Agregar divergencias al gráfico del RSI
#     for divergence in divergences:
#         if divergence[2] == 'Bullish':
#             fig.add_trace(go.Scatter(x=[divergence[0]], y=[divergence[1]], mode='markers', 
#                                      marker=dict(color='rgba(0,255,0,0.4)', size=8, symbol='arrow-up'), 
#                                      name='Bullish'),row=4,col=1)
#         if divergence[2] == 'Bearish':
#             fig.add_trace(go.Scatter(x=[divergence[0]], y=[divergence[1]], 
#                                      mode='markers', marker=dict(color='rgba(255,0,0,0.4)', size=8, symbol='arrow-down'), 
#                                      name='Bearish'),row=4,col=1)
#     
#     if show_simple_trade: #Bollinger Bands
#         fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_High'], mode='lines', name='Bollinger High',
#                                  line=dict(color='rgba(248, 237, 98, 0.3)')))
#
#         fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Low'], mode='lines', name='Bollinger Low',
#                                  line=dict(color='rgba(233,215,0, 0.3)')))  # Color similar a rgba(214, 39, 40, 0.3)
#
#         # Añadir área sombreada entre Bollinger High y Bollinger Low
#         fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_High'], fill='tonexty',
#                                  fillcolor='rgba(248,237,98, 0.1)', line=dict(color='rgba(248, 237, 98, 0.3)'),
#                                  mode='lines', name='Bollinger Bands'))
#         fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Low'], fill='tonexty',
#                                  fillcolor='rgba(248,237,98, 0.1)', line=dict(color='rgba(214,39,40, 0.3)'),
#                                  mode='lines', name='Bollinger Bands'))
#
#
#     if show_MM:
#         fig.add_trace(go.Scatter(x=data.index, y=data['SMAII'], mode='lines', name='SMA II',
#                               line=dict(color='rgba(153,204,255, 0.8)')))
#     if show_MMI:
#         fig.add_trace(go.Scatter(x=data.index, y=data['SMAI'], mode='lines', name='SMA I', line=dict(color='rgba(85,136,255,0.8)', width=1)))
#
#         # format = '%d-%m-%Y'
#
#     # buy_signals = data[data['Buy'] == 1]
#     # sell_signals = data[data['Sell'] == 1]
#
#     # fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers+text', name='Buy Signal',
#     #                              marker=dict(color='#65fe08', size=15, symbol="arrow-up"), 
#     #                              text=buy_signals.index.strftime(format),
#     #                              textposition="bottom left", textfont=dict(color='#65fe08')))
#     #
#     # fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers+text', name='Sell Signal',
#     #                              marker=dict(color='#F93822', size=12, symbol="arrow-down"), 
#     #                              text=sell_signals.index.strftime(format),
#     #                              textposition="top right",textfont=dict(color='#F93822')))
#
#     alldays =set(data.Datetime[0]+timedelta(x) for x in range((data.Datetime[len(data.Datetime)-1]- data.Datetime[0]).days))
#     missing=sorted(set(alldays)-set(data.Datetime))
#
#     fig.update_xaxes(rangebreaks=[dict(values=missing)])
#
#     fig.update_layout(title=f'{ticker} - {company_name}', 
#                       xaxis_title='', yaxis_title='',
#                       yaxis=dict(
#                                 gridcolor='rgba(200, 200, 200, 0.03)',  # Color de la cuadrícula con transparencia
#                                 gridwidth=1  # Anchura de la cuadrícula
#                         ),
#                       xaxis_rangeslider_visible=False,
#                       height=700,
#                       dragmode='drawline',  # Habilitar el modo de dibujo de líneas
#                       shapes=[],  # Inicializar lista vacía para las líneas dibujadasleyenda, ya que solo hay dos gráficos
#                       newshape=dict(line=dict(color="red")),
#                       modebar_add=['drawline','eraseshape'],
#                       showlegend=False)
#
#     return fig

# def f_backtesting(data):
#     # DataFrame para almacenar los resultados del backtesting
#     results = []
#     old_price = 0
#     counter = 0
#     
#     # Recorrer el DataFrame `data` para identificar las operaciones
#     for index, row in data.iterrows():
#         if row['Buy'] == 1.0 or row['Sell'] == 1.0:
#             print(row['Buy'],row['Sell'])
#             buy_price = 0
#             sell_price = 0
#             trade_return = 0
#
#             if row['Buy'] == 1:
#                 buy_price = row['Close']
#                 signal_date = index
#                 action = 'Buy'
#                 if counter == 0:
#                     counter =+ 1
#                     trade_return = 0
#                 else:
#                     # Registrar la compra
#                     sell_price=old_price
#                     try:
#                         trade_return = (sell_price - buy_price) / buy_price
#                     except:
#                         trade_return = 0
#                 old_price = buy_price
#             elif row['Sell'] == 1:
#                 if counter == 0:
#                     continue
#                 # Registrar la venta
#                 buy_price=old_price
#                 sell_price = row['Close']
#                 signal_date = index
#                 action = 'Sell'
#                 old_price=sell_price
#                 # Calcular el rendimiento de la operación
#                 try:
#                     trade_return = (sell_price - buy_price) / buy_price
#                 except:
#                     trade_return = 0
#             # Añadir la operación al DataFrame de resultados
#             results.append({
#                     'Date': signal_date,
#                     'Action' : action,
#                     'Buy_Price': buy_price,
#                     'Sell_Price': sell_price,
#                     'Return': trade_return
#                 })
#     return pd.DataFrame(results)
#
# # Función para aplicar formato condicional y otros estilos
# def style_dataframe(df):
#     df['Return_Percent'] = df['Return'] * 100  # Convertir a porcentaje para barras de progreso
#     df['Return'] = df['Return'].apply(lambda x: "{:.2%} 💰️".format(x))  # Formatear como porcentaje
#
#     styled_df = df.style.applymap(
#         lambda x: 'color: red;' if isinstance(x, str) and '-' in x else 'color: green;' if isinstance(x, str) else '',
#         subset=['Return'])
#
#     return styled_df

def plot_data(data, ticker, show_g_channel, show_simple_trade, show_MM, show_MMI, show_par=True):
    # Asegurar que la columna 'Datetime' esté en formato datetime
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    
    format = '%Y-%m-%d %H:%M'
    company_name = get_company_name(ticker)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        row_heights=[0.7, 0.1, 0.1, 0.1],
                        vertical_spacing=0.05)

    # Añadir gráfico de velas (candlestick)
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'), row=1, col=1)

    if show_g_channel:
        # Calcular POC, VAL y VAH
        poc_price, val, vah = calculate_poc_val_vah(data)
        
        # Identificar otros máximos relativos en el volumen
        volume_peaks = data['Volume'][(data['Volume'].shift(2) < data['Volume']) & (data['Volume'].shift(-2) < data['Volume'])]

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
                line=dict(color='rgba(93,93,93,0.1)', dash='dot')
            )))

    if show_par:
        fig.add_trace(go.Scatter(x=data.index, y=data['SAR'],
                             mode='markers',
                             marker=dict(color='rgba(125,132,113,0.4)', size=5),
                             name='Parabolic SAR'))
    

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
                             marker_color='rgba(131, 119, 180, 0.6)'),
                  row=4, col=1)
    fig.add_trace(go.Scatter(
        mode='lines',
                y=[30, 30],
                x=[data.index[0], data.index[-1]],
                             marker_color='rgba(239,169,74,0.3)'),
                  row=4, col=1)
    fig.add_trace(go.Scatter(
                mode='lines',
                y=[70, 70],
                x=[data.index[0], data.index[-1]],
                marker_color='rgba(239,169,74,0.3)'),
                row=4, col=1)

    divergences = identify_rsi_divergences(data)
    for divergence in divergences:
        if divergence[2] == 'Bullish':
            fig.add_trace(go.Scatter(x=[divergence[0]], y=[divergence[1]], mode='markers', 
                                     marker=dict(color='rgba(0,255,0,0.4)', size=8, symbol='arrow-up'), 
                                     name='Bullish'),row=4,col=1)
        if divergence[2] == 'Bearish':
            fig.add_trace(go.Scatter(x=[divergence[0]], y=[divergence[1]], 
                                     mode='markers', marker=dict(color='rgba(255,0,0,0.4)', size=8, symbol='arrow-down'), 
                                     name='Bearish'),row=4,col=1)
    
    if show_simple_trade:
        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_High'], mode='lines', name='Bollinger High',
                                 line=dict(color='rgba(248, 237, 98, 0.3)')))

        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Low'], mode='lines', name='Bollinger Low',
                                 line=dict(color='rgba(233,215,0, 0.3)')))

        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_High'], fill='tonexty',
                                 fillcolor='rgba(248,237,98, 0.1)', line=dict(color='rgba(248, 237, 98, 0.3)'),
                                 mode='lines', name='Bollinger Bands'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Low'], fill='tonexty',
                                 fillcolor='rgba(248,237,98, 0.1)', line=dict(color='rgba(214,39,40, 0.3)'),
                                 mode='lines', name='Bollinger Bands'))

    if show_MM:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMAII'], mode='lines', name='SMA II',
                              line=dict(color='rgba(153,204,255, 0.8)')))
    if show_MMI:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMAI'], mode='lines', name='SMA I', line=dict(color='rgba(85,136,255,0.8)', width=1)))

    # Encontrar días faltantes
    alldays = set(data['Datetime'].min() + timedelta(days=x) for x in range((data['Datetime'].max() - data['Datetime'].min()).days + 1))
    missing = sorted(set(alldays) - set(data['Datetime']))

    # Configurar los rangebreaks en los ejes x
    fig.update_xaxes(rangebreaks=[dict(values=missing)])

    fig.update_layout(title=f'{ticker} - {company_name}', 
                      xaxis_title='', yaxis_title='',
                      yaxis=dict(
                                gridcolor='rgba(200, 200, 200, 0.03)',  # Color de la cuadrícula con transparencia
                                gridwidth=1  # Anchura de la cuadrícula
                        ),
                      xaxis_rangeslider_visible=False,
                      height=700,
                      dragmode='drawline',  # Habilitar el modo de dibujo de líneas
                      shapes=[],  # Inicializar lista vacía para las líneas dibujadas
                      newshape=dict(line=dict(color="red")),
                      modebar_add=['drawline','eraseshape'],
                      showlegend=False)

    return fig
