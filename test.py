import pandas as pd

def crear_barra(porcentaje, max_longitud=80):
    longitud_llena = int((porcentaje / 100) * max_longitud)
    barra = '█' * longitud_llena + ' ' * (max_longitud - longitud_llena)
    return barra

def obtener_constituyentes_y_pesos():

    headers = {"User-Agent": "Mozilla/5.0"}
    # URL para obtener los constituyentes y pesos del S&P 500
    url = 'https://www.slickcharts.com/sp500'

    # Leer la tabla desde la URL usando pandas
    try:
        tablas = pd.read_html(url, storage_options=headers)
#        print(tablas)
    except ValueError as e:
        print(f"Error al leer la tabla desde la URL: {e}")
        return None

    # La primera tabla generalmente contiene los datos de interés
    df = tablas[0].set_index('#')
    df['weight'] = df['Portfolio%'].str.rstrip('%').astype('float')

    df['Barra'] = df['weight'].apply(lambda x: crear_barra(x))

    return df[['Company','Symbol','weight','Barra']]
