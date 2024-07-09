import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import xgboost as xgb
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from calculations.calculations import (
repulsion_alisada, tema, dema, calculate_cmf,
calculate_moving_average, normalize_sma_to_range, 
normalize_cmf_to_range, calculate_rsi, calculate_macd,
get_mann_kendall, calculate_ahimud, calcular_fibonacci
)

# Configuración global de tamaño de fuente para matplotlib
mpl.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.size': 10})

plt.style.use("dark_background")

all_margins=dict(l=20, r=20, t=30, b=0)

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

def plot_price_and_volume(data_in, markers, full_data):
    data = data_in.copy()
    # Convertir el índice a datetime si es necesario
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.astype(str)

    # Crear un objeto de figura de Plotly con subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                        row_heights=[0.8, 0.2],
                        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]  # Añadir un segundo eje Y solo al subplot inferior
                        )  # 2 filas, 1 columna
    # row_heights=[0.7, 0.3] significa que la primera fila (precio) será 0.7 veces más alta que la segunda fila (volumen)

    # Añadir el gráfico del precio (arriba)
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='dodgerblue', width=2)), row=1, col=1)

    # Añadir texto de máximo y mínimo para el gráfico de precios
    # max_price = data['Close'].max()
    # min_price = data['Close'].min()
    # idx_max = data['Close'].idxmax()
    # idx_min = data['Close'].idxmin()
    # fig.add_annotation(x=idx_max, y=max_price, text=f'Max: {max_price:.3f}', showarrow=True, arrowhead=1, ax=0, ay=-40, row=1, col=1)
    # fig.add_annotation(x=idx_min, y=min_price, text=f'Min: {min_price:.3f}', showarrow=True, arrowhead=1, ax=0, ay=40, row=1, col=1)

    # Añadir el gráfico de volumen (abajo)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker=dict(color='rgba(31,119,180,0.6)')), row=2, col=1)
    # Agregar el indicador de Amihud al subplot inferior
    if markers['liquidity']:
        data = calculate_ahimud(data)
        fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Amihud'],
            mode='lines',
            name='Amihud',
            line=dict(color='rgba(0, 255, 0, 0.2)'),
            yaxis="y2"
        ),
        row=2, col=1,
        secondary_y=True
    )
    if markers['kendall']:
        tau, p_value, trend = get_mann_kendall(data)
            # Añadir anotaciones para mostrar el resultado de la prueba de Mann-Kendall
        if trend == 'Bearish':
            fig.add_annotation(
                x=0.05, y=0.95,
                xref='paper', yref='paper',
                text=f"Coeficiente de Kendall: {tau:.2f}<br>Tendencia: {trend}<br>P-valor: {p_value:.2f}",
                showarrow=False,
                font=dict(size=12, color='#A45A52'),
                align='left'
            )
        else:
            fig.add_annotation(
                x=0.05, y=0.95,
                xref='paper', yref='paper',
                text=f"Coeficiente de Kendall: {tau:.2f}<br>Tendencia: {trend}<br>P-valor: {p_value:.2f}",
                showarrow=False,
                font=dict(size=12, color='#29AB87'),
                align='left'
            )

    if markers['bollinger']:
        plot_bollinger_bands(data, fig)

    if markers['fibonacci']:
        # Agregar líneas para los niveles de Fibonacci
        fib_levels, fibonacci_values = calcular_fibonacci(data)
        # Agregar líneas para los niveles de Fibonacci con colores
        colors = ['blue', 'green', 'red', 'red', 'green', 'gray']
        for i, (level, value) in enumerate(zip(fib_levels, fibonacci_values)):
            fig.add_shape(type="line",
                  x0=data.index[0], y0=value, x1=data.index[-1], y1=value,
                  line=dict(color=colors[i], width=1, dash="dash"),
                  opacity=0.7,
                  name=f'Fib {level}')
    if markers['SMA200']:
        full_data['SMA200'] = full_data['Close'].rolling(window=200).mean()
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], mode='lines', line=dict(color='darkblue', width=2)), row=1, col=1)
    if markers['SMA5']:
        data['SMA5'] = data['Close'].rolling(window=5).mean()
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA5'], mode='lines', line=dict(color='cyan', width=2)), row=1, col=1)

    # Configuraciones de diseño y estilo para el gráfico completo
    fig.update_layout(
        height=600,
        margin=all_margins,
        hovermode='x',  # Activar el modo hover
        showlegend=False,  # Ocultar la     
        dragmode='drawline',  # Habilitar el modo de dibujo de líneas
        shapes=[],  # Inicializar lista vacía para las líneas dibujadasleyenda, ya que solo hay dos gráficos
        xaxis=dict(
            domain=[0, 1],  # Ajustar la posición horizontal del eje x
        ),
        yaxis=dict(
            titlefont=dict(color='rgba(31,119,180,0.6)'),
            tickfont=dict(color='rgba(31,119,180,0.6)'),
        ),
        newshape=dict(line=dict(color="red")),
        modebar_add=['drawline','eraseshape']
    )

    # Configuraciones de ejes para cada subplot
    fig.update_yaxes(title_text="", row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text="", row=2, col=1, showticklabels=False)

    return fig

def plot_candlestick(data_in, markers):
    data = data_in.copy()
    # Convertir el índice a datetime si es necesario
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.astype(str)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.8, 0.2])  # 2 filas, 1 columna

    candlestick = go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'])

    # Agregar el indicador de Amihud al subplot inferior
    if markers['liquidity']:
        data = calculate_ahimud(data)
        fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Amihud'],
            mode='lines',
            name='Amihud',
            line=dict(color='rgba(0, 255, 0, 0.2)'),
            yaxis="y2"
        ),
        row=2, col=1,
        secondary_y=True
    )
    if markers['kendall']:
        tau, p_value, trend = get_mann_kendall(data)
            # Añadir anotaciones para mostrar el resultado de la prueba de Mann-Kendall
        if trend == 'Bearish':
            fig.add_annotation(
                x=0.05, y=0.95,
                xref='paper', yref='paper',
                text=f"Coeficiente de Kendall: {tau:.2f}<br>Tendencia: {trend}<br>P-valor: {p_value:.2f}",
                showarrow=False,
                font=dict(size=12, color='#A45A52'),
                align='left'
            )
        else:
            fig.add_annotation(
                x=0.05, y=0.95,
                xref='paper', yref='paper',
                text=f"Coeficiente de Kendall: {tau:.2f}<br>Tendencia: {trend}<br>P-valor: {p_value:.2f}",
                showarrow=False,
                font=dict(size=12, color='#29AB87'),
                align='left'
            )

    if markers['bollinger']:
        plot_bollinger_bands(data, fig)

    if markers['fibonacci']:
        # Agregar líneas para los niveles de Fibonacci
        fib_levels, fibonacci_values = calcular_fibonacci(data)
        # Agregar líneas para los niveles de Fibonacci con colores
        colors = ['blue', 'green', 'red', 'red', 'green', 'gray']
        for i, (level, value) in enumerate(zip(fib_levels, fibonacci_values)):
            fig.add_shape(type="line",
                  x0=data.index[0], y0=value, x1=data.index[-1], y1=value,
                  line=dict(color=colors[i], width=1, dash="dash"),
                  opacity=0.7,
                  name=f'Fib {level}')

    if markers['SMA200']:
        data['SMA200'] = data['Close'].rolling(window=200).mean()
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], mode='lines', line=dict(color='darkblue', width=2)), row=1, col=1)

    if markers['SMA5']:
        data['SMA5'] = data['Close'].rolling(window=5).mean()
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA5'], mode='lines', line=dict(color='cyan', width=2)), row=1, col=1)

    fig.add_trace(candlestick, row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker=dict(color='rgba(31,119,180,0.6)')), row=2, col=1)

    if markers['bollinger']:
        plot_bollinger_bands(data, fig)

    # Configuraciones de diseño y estilo
    fig.update_layout(
        height=600,
        margin=all_margins,
        hovermode='x',  # Activar el modo hover
        showlegend=False,  # Ocultar la leyenda, ya que solo hay un gráfico
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            domain=[0, 1],  # Ajustar la posición horizontal del eje x
        ),
        yaxis=dict(
            titlefont=dict(color='rgba(31,119,180,0.6)'),
            tickfont=dict(color='rgba(31,119,180,0.6)'),
        ),
    )

    # Configuraciones de ejes para cada subplot
    fig.update_yaxes(title_text="", row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text="", row=2, col=1, showticklabels=False)

    return fig

def plot_ma(data_in, check_list):
    data = data_in.copy()
    # Convertir el índice a datetime si es necesario
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.astype(str)
    # Calcular medias móviles
    # fig, ax = plt.subplots(figsize=(12, 6))
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()

    # Calcular media móvil exponencial (EMA)
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.8, 0.2])  # 2 filas, 1 columna
    
    if check_list[0]:
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Real', line=dict(color='dodgerblue', width=1)), row=1, col=1)
    if check_list[2]:
        fig.add_trace(go.Scatter(x=data.index, y=data['MA_20'], mode='lines', name='MA 20', line=dict(color='orange', width=2)), row=1, col=1)
    if check_list[3]:
        fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='MA 50', line=dict(color='green', width=2)), row=1, col=1)
    if check_list[4]:
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_10'], mode='lines', name='EMA 10', line=dict(color='yellow', width=2)), row=1, col=1)

    # Calcular Holt-Winters - Aditivo
    if check_list[1]:
        try:
            model = ExponentialSmoothing(data['Close'], trend='add', seasonal='add', seasonal_periods=12)
            fitted = model.fit()
            data['Holt_Winters'] = fitted.fittedvalues
            fig.add_trace(go.Scatter(x=data.index, y=data['Holt_Winters'], mode='lines', name='Holt-Winters', line=dict(color='magenta', width=2)), row=1, col=1)
        except:
            print('error')

    # Añadir el gráfico de volumen (abajo)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker=dict(color='rgba(31,119,180,0.6)')), row=2, col=1)

    # Configuraciones de diseño y estilo para el gráfico completo
    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=80, b=0),
        hovermode='x',  # Activar el modo hover
        showlegend=True,  # Ocultar la leyenda, ya que solo hay dos gráficos
        xaxis=dict(
            domain=[0, 1],  # Ajustar la posición horizontal del eje x
        ),
        yaxis=dict(
            titlefont=dict(color='rgba(31,119,180,0.6)'),
            tickfont=dict(color='rgba(31,119,180,0.6)'),
        ),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    )
    )

    # Configuraciones de ejes para cada subplot
    fig.update_yaxes(title_text="", row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text="", row=2, col=1, showticklabels=False)

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

def plot_cmf_with_moving_averages(data, cmf_period=8, ma_period1=5, ma_period2=20):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

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
    fig.add_trace(go.Bar(x=data.index, y=norm_cmf, marker_color=np.where(cmf >= 0, 'green', 'red'), opacity=0.3, name='CFD(8)'))
    
    # Graficar Medias Móviles Normalizadas
    fig.add_trace(go.Scatter(x=data.index, y=norm_ma1, name=f'SMA {ma_period1}', line=dict(color='dodgerblue')))
    fig.add_trace(go.Scatter(x=data.index, y=norm_ma2, name=f'SMA {ma_period2}', line=dict(color='rosybrown')))

    # Personalizar el gráfico
    # Configuraciones de diseño y estilo para el gráfico completo
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=80, b=0),
        hovermode='x',  # Activar el modo hover
        showlegend=True,  # Ocultar la leyenda, ya que solo hay dos gráficos
        xaxis=dict(
            domain=[0, 1],  # Ajustar la posición horizontal del eje x
        ),
        yaxis=dict(
            titlefont=dict(color='rgba(31,119,180,0.6)'),
            tickfont=dict(color='rgba(31,119,180,0.6)'),
        ),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    )
    )
    #fig.update_layout(showlegend=True, barmode='overlay', xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="CMF & SMA", showticklabels=False)
    
    return fig

def plot_with_indicators(data):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Calcular y graficar la Repulsión Alisada
    repulsion = repulsion_alisada(data['Close'], span=5)
    fig.add_trace(go.Scatter(x=data.index, y=repulsion, name='Repulsión Alisada (5)', line=dict(color='dodgerblue')))

    # Calcular y graficar la TEMA
    tema_line = tema(data['Close'], window=21)
    fig.add_trace(go.Scatter(x=data.index, y=tema_line, name='TEMA (21)', line=dict(color='orange', dash='dash')))

    # Calcular y graficar la DEMA
    dema_line = dema(data['Close'], window=21)
    fig.add_trace(go.Scatter(x=data.index, y=dema_line, name='DEMA (21)', line=dict(color='red', dash='dash')))

    # Rellenar el área según las condiciones
    fig.add_trace(go.Scatter(x=data.index, y=repulsion, fill='tonexty', name='Área Azul', fillcolor='rgba(0,0,255,0.3)'))

    # Configuraciones de diseño y estilo para el gráfico completo
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=80, b=0),
        hovermode='x',  # Activar el modo hover
        showlegend=True,  # Ocultar la leyenda, ya que solo hay dos gráficos
        xaxis=dict(
            domain=[0, 1],  # Ajustar la posición horizontal del eje x
        ),
        yaxis=dict(
            titlefont=dict(color='rgba(31,119,180,0.6)'),
            tickfont=dict(color='rgba(31,119,180,0.6)'),
        ),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    )
    )

    fig.update_yaxes(title_text="Paddle Traders", showticklabels=False)
    
    return fig

def plot_indicators_rsi(data):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # Plot RSI
    rsi = calculate_rsi(data)
    fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[data.index[0], data.index[-1]], y=[70, 70], mode='lines', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[data.index[0], data.index[-1]], y=[30, 30], mode='lines', line=dict(color='green', dash='dash')), row=1, col=1)

    fig.update_yaxes(title_text="RSI", row=1, col=1, showticklabels=False)

    fig.update_layout(
        height=200,
        margin=all_margins,
        hovermode='x',  # Activar el modo hover
        showlegend=False,  # Ocultar la leyenda, ya que solo hay dos gráficos
        xaxis=dict(
            domain=[0, 1],  # Ajustar la posición horizontal del eje x
        ),
        yaxis=dict(
            titlefont=dict(color='rgba(31,119,180,0.6)'),
            tickfont=dict(color='rgba(31,119,180,0.6)'),
        ),
    )

    return fig

def plot_indicators_macd(data):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # Plot MACD
    macd_line, signal_line = calculate_macd(data)
    fig.add_trace(go.Scatter(x=data.index, y=macd_line, name='MACD 12/26', line=dict(color='dodgerblue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=signal_line, name='Signal', line=dict(color='magenta', dash='dash')), row=1, col=1)

    # Plot Histograma del MACD
    macd_histogram = macd_line - signal_line
    fig.add_trace(go.Bar(x=data.index, y=macd_histogram, marker_color=np.where(macd_histogram >= 0, 'green', 'darkgray'), opacity=0.6), row=1, col=1)

    fig.update_yaxes(title_text="MACD 12/26", showticklabels=False)

    fig.update_layout(
        height=200,
        margin=all_margins,
        hovermode='x',  # Activar el modo hover
        showlegend=False,  # Ocultar la leyenda, ya que solo hay dos gráficos
        xaxis=dict(
            domain=[0, 1],  # Ajustar la posición horizontal del eje x
        ),
        yaxis=dict(
            titlefont=dict(color='rgba(31,119,180,0.6)'),
            tickfont=dict(color='rgba(31,119,180,0.6)'),
        ),
    )

    return fig

def plot_volatility(df_vol):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=df_vol.index, y=df_vol['returns'], name='Returns', line=dict(color='rgba(30, 144, 255, 0.2)')))
    fig.add_trace(go.Scatter(x=df_vol.index, y=df_vol['volatility'], name='Volatility', line=dict(color='darkorange')))

    max_vol = df_vol['volatility'].max()
    min_vol = df_vol['volatility'].min()
    idx_max = df_vol['volatility'].idxmax()
    idx_min = df_vol['volatility'].idxmin()

    fig.add_annotation(x=idx_max, y=max_vol, text=f'Max: {max_vol:.3f}', showarrow=True, arrowhead=1)
    fig.add_annotation(x=idx_min, y=min_vol, text=f'Min: {min_vol:.3f}', showarrow=True, arrowhead=1)

    # Añadir líneas horizontales discontinuas
    fig.add_shape(
        type="line",
        x0=df_vol.index.min(), y0=0.01, x1=df_vol.index.max(), y1=0.01,
        line=dict(color="red", width=2, dash="dashdot")
    )

    fig.add_shape(
        type="line",
        x0=df_vol.index.min(), y0=0, x1=df_vol.index.max(), y1=0,
        line=dict(color="lightgray", width=2, dash="dashdot")
    )

    fig.add_shape(
        type="line",
        x0=df_vol.index.min(), y0=0.005, x1=df_vol.index.max(), y1=0.005,
        line=dict(color="green", width=2, dash="dashdot")
    )

    # Configuraciones de diseño y estilo para el gráfico completo
    fig.update_layout(
        height=200,
        margin=all_margins,
        hovermode='x',  # Activar el modo hover
        showlegend=False,
        legend=dict(x=0.05, y=0.95, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='rgba(0, 0, 0, 0.5)'),  # Mostrar leyenda
        xaxis=dict(
            domain=[0, 1],  # Ajustar la posición horizontal del eje x
        ),
        yaxis=dict(
            title="Volatility",
            titlefont=dict(color='rgba(31,119,180,0.6)'),
            tickfont=dict(color='rgba(31,119,180,0.6)'),
            showticklabels=True,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='white'
        ),
    )
    fig.update_yaxes(title_text="Volatility", showticklabels=False)

    return fig

def plot_bollinger_bands(df, fig, window=20, num_std_dev=2):
    """
    Plotea las Bandas de Bollinger a partir de un DataFrame que contiene los datos históricos.

    :param df: DataFrame que contiene los datos históricos con columnas ['Open', 'High', 'Low', 'Close'].
    :param window: Ventana de tiempo para calcular la media móvil simple (SMA).
    :param num_std_dev: Número de desviaciones estándar para calcular las bandas superior e inferior.
    :return: Figura de Plotly con el gráfico de velas y las Bandas de Bollinger.
    """
    
    # Calcular la Media Móvil Simple (SMA)
    df['SMA'] = df['Close'].rolling(window=window).mean()
    
    # Calcular la Desviación Estándar
    df['STD'] = df['Close'].rolling(window=window).std()
    
    # Calcular las Bandas de Bollinger
    df['Upper Band'] = df['SMA'] + (df['STD'] * num_std_dev)
    df['Lower Band'] = df['SMA'] - (df['STD'] * num_std_dev)
    
    # Añadir la media móvil
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA'],
        line=dict(color='orange', width=2),
        name='SMA'
    ))

    # Añadir las bandas de Bollinger
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Upper Band'],
        line=dict(color='red', width=1),
        name='Upper Band',
        opacity=0.5
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Lower Band'],
        line=dict(color='green', width=1),
        name='Lower Band',
        opacity=0.5
    ))

    # Rellenar el área entre las bandas
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Upper Band'],
        line=dict(color='red', width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Lower Band'],
        line=dict(color='rgba(0,100,80,0.2)', width=0),
        fill='tonexty',
        showlegend=False
    ))

def plot_per_gauge(per_actual):
    # Definir los rangos para la interpretación del PER
    valoracion_baja = 10
    valoracion_media = 20
    valoracion_alta = 30
    
    # Crear el gráfico gauge
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = per_actual,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Price-to-Earnings Ratio (PER)"},
        gauge = {
            'axis': {'range': [None, valoracion_alta + 5]},
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [0, valoracion_baja], 'color': "lightgreen"},
                {'range': [valoracion_baja, valoracion_media], 'color': "yellow"},
                {'range': [valoracion_media, valoracion_alta], 'color': "orange"},
                {'range': [valoracion_alta, valoracion_alta + 5], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': per_actual}
        }
    ))

    return fig
