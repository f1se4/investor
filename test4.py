import requests

# URL de la API para obtener los componentes del S&P 500
url = 'https://financialmodelingprep.com/api/v3/sp500_constituent'

# Parámetros opcionales si deseas filtrar o modificar la respuesta
params = {
    'apikey': 'your_api_key'  # Reemplaza 'your_api_key' con tu clave API de Financial Modeling Prep si es necesario
}

# Realizar la solicitud GET a la API
response = requests.get(url)

# Verificar si la solicitud fue exitosa (código de estado 200)
if response.status_code == 200:
    # Convertir la respuesta JSON en un diccionario Python
    data = response.json()
    
    # Extraer la lista de componentes del S&P 500
    sp500_components = data['symbolsList']
    
    # Mostrar los nombres de los componentes y tickers
    for component in sp500_components:
        print(component['symbol'], '-', component['name'])
else:
    print(f'Error al obtener los componentes del S&P 500. Código de estado: {response.status_code}')

