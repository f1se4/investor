import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts
import numpy as np
import json

COLOR_BULL = 'rgba(38,166,154,0.9)' # #26a69a
COLOR_BEAR = 'rgba(239,83,80,0.9)'  # #ef5350

def f_daily_plot(df):
    # Some data wrangling to match required format
    df.columns = ['time','open','high','low','close','volume']
    df['time'] = df['time'].view('int64') // 10**9
    df['color'] = np.where(  df['open'] > df['close'], COLOR_BEAR, COLOR_BULL)  # bull or bear

    # export to JSON format
    candles = json.loads(df.to_json(orient = "records"))
    volume = json.loads(df[['time','volume']].rename(columns={"volume": "value",}).to_json(orient = "records"))
    priceVolumeChartOptions = {
        "height": 400,
        "layout": {
            "background": {
                "type": "solid",
                "color": 'transparent'
            },
            "textColor": "white"
        },
        "grid": {
            "vertLines": {
                "color": 'rgba(42, 46, 57, 0)',
            },
            "horzLines": {
                "color": 'rgba(42, 46, 57, 0.6)',
            }
        },
        "crosshair": {
            "mode": 0
        },
        "timeScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)",
            "barSpacing": 10,
            "minBarSpacing": 8,
            "timeVisible": True,
            "secondsVisible": False,
        },
    }

    priceVoulumeSeries = [
    {
        "type": 'Candlestick',
        "data": candles,
        "options": {
            "upColor": COLOR_BULL,
            "downColor": COLOR_BEAR,
            "borderVisible": False,
            "wickUpColor": COLOR_BULL,
            "wickDownColor": COLOR_BEAR
        }
    },
    {
        "type": 'Histogram',
        "data": volume,
        "options": {
            "color": 'rgba(38, 166, 154, 0.5)',
            "priceFormat": {
                "type": 'volume',
            },
            "priceScaleId": "" # set as an overlay setting,
        },
        "priceScale": {
            "scaleMargins": {
                "top": 0.7,
                "bottom": 0,
            }
        }
    }
]
    return renderLightweightCharts([
    {
        "chart": priceVolumeChartOptions,
        "series": priceVoulumeSeries
    }
], 'priceAndVolume')
