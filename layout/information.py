import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
from yahooquery import Ticker

def obtener_pesos_constrituyentes(index_ticker):
    """
    Función para obtener los pesos de los constituyentes de un índice dado.
    
    Parámetros:
    index_ticker (str): Símbolo del índice (por ejemplo, "^GSPC" para el S&P 500).

    Retorna:
    DataFrame: Un DataFrame con los nombres y pesos de los constituyentes.
    """
    try:
        ticker = yf.Ticker(index_ticker)
        # Obtener los constituyentes
        components = ticker.constituents
        
        # Verificar si se obtuvieron los datos
        if components is not None:
            components_df = pd.DataFrame(components)
            st.dataframe(components_df)
        else:
            st.error("No se pudo obtener la información del índice. Verifica el símbolo.")
            return None
    except Exception as e:
        st.error(f"Error al obtener los datos: {e}")
        return None

def obtener_pesos_constrituyentes(index_ticker):
    """
    Función para obtener los pesos de los constituyentes de un índice dado.
    
    Parámetros:
    index_ticker (str): Símbolo del índice (por ejemplo, "^GSPC" para S&P 500).

    Retorna:
    DataFrame: Un DataFrame con los nombres y pesos de los constituyentes.
    """
    try:
        # Obtener datos de Yahoo Finance
        components = pdr.get_components(index_ticker)
        
        # Verificar si se obtuvieron los datos
        if components is not None:
            st.dataframe(components)
        else:
            st.error("No se pudo obtener la información del índice. Verifica el símbolo.")
            return None
    except Exception as e:
        st.error(f"Error al obtener los datos: {e}")
        return None

def obtener_pesos_constrituyentes(index_ticker):
    """
    Función para obtener los pesos de los constituyentes de un índice dado.
    
    Parámetros:
    index_ticker (str): Símbolo del índice (por ejemplo, "^GSPC" para S&P 500).

    Retorna:
    DataFrame: Un DataFrame con los nombres y pesos de los constituyentes.
    """
    try:
        # Obtener los datos del índice
        index_data = yf.download(index_ticker, period="1d")
        
        # Extraer los constituyentes y pesos
        components = index_data['Weighting']
        
        # Verificar si se obtuvieron los datos
        if components is not None:
            return components.reset_index()
        else:
            st.error("No se pudo obtener la información del índice. Verifica el símbolo.")
            return None
    except Exception as e:
        st.error(f"Error al obtener los datos: {e}")
        return None

def obtener_pesos_constrituyentes(index_ticker):
    """
    Función para obtener los pesos de los constituyentes de un índice dado.
    
    Parámetros:
    index_ticker (str): Símbolo del índice (por ejemplo, "^GSPC" para S&P 500).

    Retorna:
    DataFrame: Un DataFrame con los nombres y pesos de los constituyentes.
    """
    try:
        # Crear un objeto Ticker con el símbolo del índice
        ticker = Ticker(index_ticker)
        
        # Obtener los constituyentes y sus pesos
        constituents = ticker.fund_composition
        if constituents is not None:
            # Convertir a DataFrame
            df = pd.DataFrame(constituents)
            st.dataframe(df)
        else:
            st.error("No se pudo obtener la información del índice. Verifica el símbolo.")
            return None
    except Exception as e:
        st.error(f"Error al obtener los datos: {e}")
        return None
