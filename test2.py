from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd

# Configurar Selenium y opciones de Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ejecución en modo headless (sin ventana de navegador)
service = Service('path_to_chromedriver')  # Reemplaza 'path_to_chromedriver' con la ubicación de tu chromedriver

# Iniciar el navegador
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL para obtener la lista de componentes del S&P 500 desde SlickCharts
url = 'https://www.slickcharts.com/sp500'

# Cargar la página
driver.get(url)

# Encontrar la tabla que contiene los componentes del S&P 500
table = driver.find_element(By.CLASS_NAME, 'table')
sp500_components = pd.read_html(table.get_attribute('outerHTML'))[0]

# Mostrar los nombres de los componentes y tickers
print(sp500_components[['Symbol', 'Company']])

# Cerrar el navegador
driver.quit()

