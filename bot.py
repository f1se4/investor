import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from calculations.calculations import get_company_name
from datetime import timedelta

def f_parabolic_SAR(df, af=0.03, max_af=0.3):
    high = df['High']
    low = df['Low']
    
    sar = df['Close'].copy()
    uptrend = True
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

def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window,  min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def identify_rsi_divergences(df, lbL=5, lbR=5, range_upper=60, range_lower=5):
    divergences = []
    df['PivotLow'] = df['RSI'].rolling(window=lbL + lbR + 1, center=True).apply(lambda x: x[lbL] if x[lbL] == min(x) else np.nan, raw=True)
    df['PivotHigh'] = df['RSI'].rolling(window=lbL + lbR + 1, center=True).apply(lambda x: x[lbL] if x[lbL] == max(x) else np.nan, raw=True)
    
    for i in range(len(df)):
        if not np.isnan(df['PivotLow'].iloc[i]):
            for j in range(i + 1, min(i + range_upper, len(df))):
                if not np.isnan(df['PivotLow'].iloc[j]) and df['Low'].iloc[j] > df['Low'].iloc[i] and df['RSI'].iloc[j] < df['RSI'].iloc[i]:
                    divergences.append((df.index[j], df['RSI'].iloc[j], 'Bullish'))
                if not np.isnan(df['PivotLow'].iloc[j]) and df['Low'].iloc[j] < df['Low'].iloc[i] and df['RSI'].iloc[j] > df['RSI'].iloc[i]:
                    divergences.append((df.index[j], df['RSI'].iloc[j], 'Hidden Bullish'))
        
        if not np.isnan(df['PivotHigh'].iloc[i]):
            for j in range(i + 1, min(i + range_upper, len(df))):
                if not np.isnan(df['PivotHigh'].iloc[j]) and df['High'].iloc[j] < df['High'].iloc[i] and df['RSI'].iloc[j] > df['RSI'].iloc[i]:
                    divergences.append((df.index[j], df['RSI'].iloc[j], 'Bearish'))
                if not np.isnan(df['PivotHigh'].iloc[j]) and df['High'].iloc[j] > df['High'].iloc[i] and df['RSI'].iloc[j] < df['RSI'].iloc[i]:
                    divergences.append((df.index[j], df['RSI'].iloc[j], 'Hidden Bearish'))
                    
    return divergences

def calculate_macd(df):
    df['EMA_12'] = ema(df['Close'], 12)
    df['EMA_26'] = ema(df['Close'], 26)
    df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD'] = df['MACD_Line'] - df['Signal_Line']
    return df

def bollinger_bands(series, window):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def get_data(ticker, selected_interval, select_period, smai=200, smaii=100):
    data = yf.download(ticker, period=select_period, interval=selected_interval)
    try:
        data.index = data.index.tz_convert('CET')
    except:
        pass

    data["Datetime"] = data.index

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

def calculate_poc_val_vah(data):
    price_volume_df = data[['Close', 'Volume']].copy()
    poc_index = data['Volume'].idxmax()
    poc_price = data.loc[poc_index, 'Close']
    
    price_volume_df = price_volume_df.sort_values(by='Close')
    total_volume = price_volume_df['Volume'].sum()
    target_volume = total_volume * 0.70
    
    cumulative_volume = 0
    val = None
    vah = None
    middle_price = poc_price

    lower_half_volume = price_volume_df[price_volume_df['Close'] <= middle_price]
    upper_half_volume = price_volume_df[price_volume_df['Close'] > middle_price]

    lower_cumulative_volume = lower_half_volume['Volume'].sum()
    upper_cumulative_volume = upper_half_volume['Volume'].sum()

    if lower_cumulative_volume > target_volume / 2:
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

def calculate_rangebreaks(data, interval):
    rangebreaks = []
    
    # Determinar si hay fines de semana presentes en los datos
    days_present = data.index.dayofweek.unique()
    has_weekends = any(day in days_present for day in [5, 6])
    
    # Agregar rangebreaks para fines de semana si no están presentes
    if not has_weekends:
        rangebreaks.append(dict(bounds=["sat", "mon"]))
    
    # Si el intervalo es menor a 1 día, agregar rangebreaks para horas no bursátiles
    if interval in ['1h', '30m', '15m', '5m', '1m']:
        # Obtener la primera fecha y su rango horario
        first_day = data.index[0].date()
        first_day_data = data[data.index.date == first_day]
        market_open = first_day_data.index.min().time()
        print(market_open)
        market_close = first_day_data.index.max().time()
        print(market_close)
        
        # Convertir las horas a minutos para facilitar los cálculos
        open_in_minutes = market_open.hour - 1
        print(open_in_minutes)
        close_in_minutes = market_close.hour + 1
        print(close_in_minutes)
        
        rangebreaks.append(dict(bounds=[close_in_minutes, open_in_minutes], pattern="hour"))

    return rangebreaks

def plot_data(data, ticker, interval, show_g_channel, show_simple_trade, show_MM, show_MMI, show_par=True):
    
    company_name = get_company_name(ticker)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        row_heights=[0.7, 0.1, 0.1, 0.1],
                        vertical_spacing=0.05)

    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'), row=1, col=1)

    if show_g_channel:
        poc_price, val, vah = calculate_poc_val_vah(data)
        volume_peaks = data['Volume'][(data['Volume'].shift(2) < data['Volume']) & (data['Volume'].shift(-2) < data['Volume'])]

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

    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='rgba(31, 119, 180, 0.3)'),
                  row=2, col=1)
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
        # if divergence[2] == 'Hidden Bullish':
        #     fig.add_trace(go.Scatter(x=[divergence[0]], y=[divergence[1]], mode='markers', 
        #                              marker=dict(color='rgba(0,150,0,0.4)', size=8, symbol='arrow-up'), 
        #                              name='Hidden Bullish'),row=4,col=1)
        # if divergence[2] == 'Hidden Bearish':
        #     fig.add_trace(go.Scatter(x=[divergence[0]], y=[divergence[1]], 
        #                              mode='markers', marker=dict(color='rgba(150,0,0,0.4)', size=8, symbol='arrow-down'), 
        #                              name='Hidden Bearish'),row=4,col=1)
    
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


    fig.update_xaxes(rangebreaks=calculate_rangebreaks(data, interval))

    fig.update_layout(title=f'{ticker} - {company_name}', 
                      xaxis_title='', yaxis_title='',
                      yaxis=dict(
                                gridcolor='rgba(200, 200, 200, 0.03)',
                                gridwidth=1
                        ),
                      xaxis_rangeslider_visible=False,
                      height=700,
                      dragmode='drawline',
                      shapes=[],
                      newshape=dict(line=dict(color="red")),
                      modebar_add=['drawline','eraseshape'],
                      showlegend=False)

    return fig
