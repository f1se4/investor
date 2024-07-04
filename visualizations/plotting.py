import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mplfinance as mpf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import xgboost as xgb
from datetime import timedelta
import plotly.tools as tls
import plotly.graph_objects as go
import plotly.graph_objs as go2
from plotly.subplots import make_subplots

from calculations.calculations import repulsion_alisada, tema, dema, calculate_cmf, calculate_moving_average, normalize_sma_to_range, normalize_cmf_to_range, calculate_rsi, calculate_macd, get_levels

# Configuración global de tamaño de fuente para matplotlib
mpl.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.size': 10})

plt.style.use("dark_background")

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

# def plot_full(data):
#     fig, ax = plt.subplots(figsize=(14, 6))
#     
#     # Gráfico del Precio
#     ax.plot(data.index, data.Close, color='dodgerblue', linewidth=1)
#     ax.set_ylabel('')
#     ax.tick_params(axis='x', rotation=45, which='both')
#     ax.grid(True,color='gray', linestyle='-', linewidth=0.01)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#
#     ax.yaxis.set_ticks([])  # Quitar los ticks
#
#     # Añadir texto de máximo y mínimo
#     max_price = data['Close'].max()
#     min_price = data['Close'].min()
#     
#     # Encontrar el índice correspondiente al máximo y mínimo
#     idx_max = data['Close'].idxmax()
#     idx_min = data['Close'].idxmin()
#     
#     plt.text(idx_max, max_price, f'{max_price:.3f}', va='bottom', ha='center', color='dodgerblue')
#     plt.text(idx_min, min_price, f'{min_price:.3f}', va='top', ha='center', color='dodgerblue')
#
#     fig.tight_layout()
#
#     plotly_fig = tls.mpl_to_plotly(fig)
#
#     return plotly_fig

def plot_full(data_in):
    data = data_in.copy()
    # Convertir el índice a datetime si es necesario
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.astype(str)

    # Crear un objeto de figura de Plotly con subplots
    fig = make_subplots(rows=1, cols=1)

    # Añadir el gráfico del precio
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='dodgerblue', width=2)))

    # Añadir texto de máximo y mínimo
    max_price = data['Close'].max()
    min_price = data['Close'].min()
    idx_max = data['Close'].idxmax()
    idx_min = data['Close'].idxmin()

    fig.add_annotation(x=idx_max, y=max_price, text=f'Máximo: {max_price:.3f}', showarrow=True, arrowhead=1, ax=0, ay=-40)
    fig.add_annotation(x=idx_min, y=min_price, text=f'Mínimo: {min_price:.3f}', showarrow=True, arrowhead=1, ax=0, ay=40)

    # Configuraciones de diseño y estilo
    fig.update_layout(
        title='Gráfico interactivo de Precio de Cierre',
        hovermode='x',  # Activar el modo hover
        plot_bgcolor='gray',
        margin=dict(l=0, r=0, t=50, b=0),  # Ajustar márgenes
    )

    return fig

def plot_candlestick(data):
    # Extract data for candlestick chart
    candlestick = go2.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'])

    # Create figure and layout
    fig = go2.Figure(data=[candlestick])
    fig.update_layout(
        title='Candlestick Chart',
        yaxis=dict(
            tickposition='left',  # Move y-axis to the left
        ),
        xaxis=dict(
            rangeslider=dict(  # Add a range slider to zoom in/out
                visible=False
            ),
            type='date'  # Ensure x-axis is treated as date/time
        ),
        yaxis_side='left',  # Ensure y-axis ticks are on the left
        plot_bgcolor='white',  # Set plot background color
        xaxis_showgrid=False,  # Hide x-axis grid lines
        yaxis_showgrid=True,   # Show y-axis grid lines
        yaxis_gridcolor='gray',  # Set color of y-axis grid lines
        yaxis_zeroline=False,   # Hide y-axis zero line
        margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins
    )

    return fig

import plotly.graph_objs as go

def plot_candlestick(data):
    candlestick = go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'])

    fig = go.Figure(data=[candlestick])

    fig.update_layout(
        title='Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        margin=dict(l=50, r=50, t=50, b=50),
    )

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

def plot_xgboost_forecast(data, periods):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Preparar datos para el modelo (ejemplo básico)
    data['Date'] = data.index
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day

    # Variable objetivo (por ejemplo, Adj Close)
    y = data['Close']

    # Variables predictoras (ejemplo básico)
    X = data[['Year', 'Month', 'Day']]

    # Entrenar modelo xgboost
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X, y)

    # Obtener la última fecha en los datos históricos
    ultima_fecha = data.index[-1]
    ultimo_valor = data.iloc[-1]['Close']

    df_resultados = pd.DataFrame({'Fecha': [ultima_fecha], 'Prediccion': [ultimo_valor]})
    print(df_resultados)

    # Crear un DataFrame con las fechas futuras para predecir
    fechas_futuras = pd.date_range(start=ultima_fecha + timedelta(days=1), periods=periods, freq='D')
    df_prediccion = pd.DataFrame(index=fechas_futuras, columns=['Year', 'Month', 'Day'])

    # Llenar el DataFrame con valores de año, mes y día para las fechas futuras
    df_prediccion['Year'] = df_prediccion.index.year
    df_prediccion['Month'] = df_prediccion.index.month
    df_prediccion['Day'] = df_prediccion.index.day

    # Hacer la predicción con el modelo xgboost
    predicciones = model.predict(df_prediccion)

    # Crear un DataFrame con las fechas y las predicciones
    df_resultados = pd.concat([df_resultados, pd.DataFrame({'Fecha': fechas_futuras, 'Prediccion': predicciones})], ignore_index=True)

    # Graficar los resultados con Matplotlib
    ax.plot(data.index, data['Close'], color='dodgerblue', linewidth=1, alpha=0.8)
    ax.plot(df_resultados['Fecha'], df_resultados['Prediccion'], linestyle='--', color='orange', label='Forecast')
    ax.yaxis.set_ticks([])  # Quitar los ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(True, color='gray', linestyle='-', linewidth=0.01)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    ax.legend()

    return fig  # Devolvemos la figura actual de Matplotlib

def plot_rendimiento(ticker):
    # Función para obtener el rendimiento en porcentaje
    def get_performance(ticker, period):
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty or len(data) < 2:
            return None
        
        # Calcular el rendimiento porcentual
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        performance = (end_price - start_price) / start_price * 100
        
        return performance
    
    # Función para crear la figura de Plotly con los gráficos de velocímetro
    def create_gauge_fig(ticker):
        periods = ['5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        fig = go.Figure()
    
        for i, period in enumerate(periods):
            performance = get_performance(ticker, period)
            if performance is not None:
                if performance < 0:
                    color = 'darkred'
                else:
                    color = 'lightgreen'
                
                fig.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = performance,
                    title = {'text': f"{period}"},
                    gauge = {
                        'axis': {'range': [-100, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [-100, 100], 'color': 'darkslategray'},
                        ],
                        'threshold': {
                            'line': {'color': 'red', 'width': 4},
                            'thickness': 0.75,
                            'value': 0
                        }
                    },
                    domain = {'row': positions[i][0], 'column': positions[i][1]}
                ))
            else:
                st.warning(f'No hay datos disponibles para el periodo: {period}')
    
        fig.update_layout(height=500, width=1000, grid={'rows': 2, 'columns': 4, 'pattern': "independent"})
    
        return fig

    return create_gauge_fig(ticker)
