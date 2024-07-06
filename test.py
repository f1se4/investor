import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL para obtener la lista de componentes del S&P 500 desde SlickCharts
url = 'https://www.slickcharts.com/sp500'
url = 'https://es.marketscreener.com/cotizacion/indice/S-P-500-4985/componentes/'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Realizar la solicitud GET
response = requests.get(url)

# Verificar si la solicitud fue exitosa (código de estado 200)
if response.status_code == 200:
    # Parsear el contenido HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Encontrar la tabla que contiene los componentes del S&P 500
    table = soup.find('table', {'class': 'table table-hover table-borderless table-sm'})
    
    # Extraer los datos de la tabla y convertirlos en un DataFrame de pandas
    sp500_components = pd.read_html(str(table))[0]
    
    # Mostrar los nombres de los componentes y tickers
    print(sp500_components[['Symbol', 'Company']])
else:
    print(f'Error al obtener la página: {response.status_code}')

