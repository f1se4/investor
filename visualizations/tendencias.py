# Importar las bibliotecas necesarias
import numpy as np
from scipy.stats import kendalltau
import plotly.graph_objects as go
import ruptures as rpt

# Función para el modelo de cambio de régimen
def modelo_cambio_regimen(data):
    # Convertir los datos a formato necesario para ruptures
    data_array = data['Adj Close'].values.reshape(-1, 1)
    
    # Aplicar el modelo de ruptura (cambio de régimen)
    model = rpt.Pelt(model="rbf").fit(data_array)
    result = model.predict(pen=10)
    
    # Crear figura de Plotly
    fig = go.Figure()
    
    # Graficar datos de precio de cierre ajustado
    fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Precio de cierre ajustado'))
    
    # Añadir líneas verticales en los puntos de cambio
    for idx in result:
        fig.add_vline(x=data.index[idx], line_dash='dash', line_color='red',
                      annotation_text=f'Cambio de régimen', annotation_position='top left')
    
    # Añadir título y etiquetas de ejes
    fig.update_layout(title=f'Precio de cierre ajustado para {data.columns[0]}',
                      xaxis_title='Fecha',
                      yaxis_title='Precio')
    
    return fig
