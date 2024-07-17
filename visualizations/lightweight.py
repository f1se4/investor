from streamlit_lightweight_charts import renderLightweightCharts
import numpy as np
import json
import pandas as pd

COLOR_BULL = 'rgba(38,166,154,0.9)' # #26a69a
COLOR_BULL_HIST = 'rgba(57,255,20,0.3)' # #39ff14
COLOR_BEAR = 'rgba(239,83,80,0.9)'  # #ef5350
COLOR_BEAR_HIST = 'rgba(239,83,80,0.4)'  # #ef5350

def calculate_sma(df, window):
    return df['close'].rolling(window=window).mean()

# Paso 1: Importar las librerías necesarias y definir la función para calcular la volatilidad.
def calculate_volatility(df):
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=21).std() * 100  # Volatilidad en porcentaje
    return df

def calculate_micro_pullback(df):
    df['change'] = df['close'].diff()
    df['change_prev'] = df['change'].shift(1)
    micro_pullback = (df['change_prev'] < 0) & (df['change'] > 0)
    return micro_pullback

def calculate_bull_flag(df):
    df['change_prev'] = df['change'].shift(1)
    df['change_next'] = df['change'].shift(-1)
    bull_flag = (df['change_prev'] > 0) & (df['change_next'] < 0) & (df['change_prev'] > df['change_next'])
    return bull_flag

def calculate_macd(df):
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_bollinger_bands(df, window=20, num_of_std=2):
    df['sma'] = df['close'].rolling(window).mean()
    df['std'] = df['close'].rolling(window).std()
    df['upper_band'] = df['sma'] + (df['std'] * num_of_std)
    df['lower_band'] = df['sma'] - (df['std'] * num_of_std)
    return df[['time', 'upper_band', 'lower_band']]

def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rs = rs.fillna(0)
    # print(rs)
    return 100 - (100 / (1 + rs))

def f_daily_plot(df, df_sm,
                 show_sma200=False, show_sma5=False, show_macd=False, 
                 show_rsi=False, show_volatility=False, show_bollinger=False, 
                 chart_height=500):
    df = df.reset_index()
    df_sm = df_sm.reset_index()
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df_sm.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df['time'] = df['time'].view('int64') // 10**9
    df_sm['time'] = df_sm['time'].view('int64') // 10**9

    df['color'] = np.where(df['open'] > df['close'], COLOR_BEAR, COLOR_BULL)  # bull or bear

    candles = json.loads(df[['time', 'open', 'high', 'low', 'close', 'volume']].to_json(orient="records"))
    volume = json.loads(df[['time', 'volume']].rename(columns={"volume": "value"}).to_json(orient="records"))
    df['micro_pullback'] = calculate_micro_pullback(df)
    df['bull_flag'] = calculate_bull_flag(df)

    price_volume_series = [
        {
            "type": 'Candlestick',
            "data": candles,
            "options": {
                "upColor": COLOR_BULL,
                "downColor": COLOR_BEAR,
                "borderVisible": False,
                "wickUpColor": COLOR_BULL,
                "wickDownColor": COLOR_BEAR
            },
        },
        {
            "type": 'Histogram',
            "data": volume,
            "options": {
                "color": 'rgba(38, 166, 154, 0.5)',
                "priceFormat": {
                    "type": 'volume',
                },
                "priceScaleId": "" # set as an overlay setting
            },
            "priceScale": {
                "scaleMargins": {
                    "top": 0.7,
                    "bottom": 0,
                }
            }
        }
    ]
    if show_sma200:
        df_sm['sma200'] = calculate_sma(df_sm, 200)
        # Realizar el merge_asof
        df_sm = pd.merge_asof(df, df_sm, on='time', direction='nearest')

        sma200 = json.loads(df_sm[['time', 'sma200']].rename(columns={"sma200": "value"}).dropna().to_json(orient="records"))
        price_volume_series.append({
            "type": 'Line',
            "data": sma200,
            "options": {
                "color": 'rgba(255, 215, 0, 0.8)',
                "lineWidth": 2,
            }
        })
    
    if show_sma5:
        df['sma5'] = calculate_sma(df, 5)
        sma5 = json.loads(df[['time', 'sma5']].rename(columns={"sma5": "value"}).dropna().to_json(orient="records"))
        price_volume_series.append({
            "type": 'Line',
            "data": sma5,
            "options": {
                "color": 'rgba(75, 0, 130, 0.8)',
                "lineWidth": 2,
            }
        })

    if show_bollinger:
        bollinger_bands = calculate_bollinger_bands(df)
        upper_band = json.loads(bollinger_bands[['time', 'upper_band']].rename(columns={"upper_band": "value"}).dropna().to_json(orient="records"))
        lower_band = json.loads(bollinger_bands[['time', 'lower_band']].rename(columns={"lower_band": "value"}).dropna().to_json(orient="records"))
        
        price_volume_series.append({
            "type": 'Line',
            "data": upper_band,
            "options": {
                "color": 'rgba(64, 224, 208, 0.8)',
                "lineWidth": 1,
            }
        })
        price_volume_series.append({
            "type": 'Line',
            "data": lower_band,
            "options": {
                "color": 'rgba(64, 224, 208, 0.8)',
                "lineWidth": 1,
            }
        })
        price_volume_series.append({
            "type": 'Area',
            "data": upper_band,
            "options": {
                "color": 'rgba(64, 224, 208, 0.3)',
                "lineWidth": 0,
                "topColor": 'rgba(64, 224, 208, 0.3)',
                # "bottomColor": 'rgba(64, 224, 208, 0.3)',
                "bottomColor": 'transparent',
                "priceLineVisible": False,
            }
        })
        price_volume_series.append({
            "type": 'Area',
            "data": lower_band,
            "options": {
                "color": 'rgba(64, 224, 208, 0.3)',
                "lineWidth": 0,
                "topColor": 'rgba(64, 224, 208, 0.3)',
                "bottomColor": 'rgba(64, 000, 003, 0.1)',
                "priceLineVisible": False,
            }
        })

    additional_charts = []

    if show_macd:
        df['macd'], df['signal'], df['histogram'] = calculate_macd(df)
        macd = json.loads(df[['time', 'macd']].rename(columns={"macd": "value"}).dropna().to_json(orient="records"))
        signal = json.loads(df[['time', 'signal']].rename(columns={"signal": "value"}).dropna().to_json(orient="records"))
        df['color_hist'] = np.where( df['histogram'] > 0, COLOR_BULL_HIST, COLOR_BEAR_HIST) 
        histogram = json.loads(df[['time', 'histogram','color_hist']].rename(columns={"histogram": "value", "color_hist":"color"}).dropna().to_json(orient="records"))
        # print(histogram)
        macd_series = [
        {
            "type": 'Line',
            "data": macd,
            "options": {
                "color": 'rgba(0, 255, 0, 0.8)',
                "lineWidth": 1,
            }
        },
        {
            "type": 'Line',
            "data": signal,
            "options": {
                "color": 'rgba(255, 0, 0, 0.8)',
                "lineWidth": 1,
            }
        },
        {
            "type": 'Histogram',
            "data": histogram
        }
    ]
        additional_charts.append({
            "chart": {
                "height": 150,
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
                    "rightOffset": 12,
                },
            },
            "series": macd_series
        })

    if show_rsi:
        df['rsi'] = calculate_rsi(df)
        rsi = json.loads(df[['time', 'rsi']].rename(columns={"rsi": "value"}).dropna().to_json(orient="records"))
        df['high_rsi'] = 70
        df['min_rsi'] = 30
        high_rsi = json.loads(df[['time', 'high_rsi']].rename(columns={"high_rsi": "value"}).dropna().to_json(orient="records"))
        min_rsi = json.loads(df[['time', 'min_rsi']].rename(columns={"min_rsi": "value"}).dropna().to_json(orient="records"))
        rsi_series = [
            {
                "type": 'Line',
                "data": rsi,
                "options": {
                    "color": 'rgba(0, 0, 255, 0.8)',
                    "lineWidth": 1,
                }
            },
            {
                "type": 'Line',
                "data": min_rsi,
                "options": {
                    "color": 'darkgreen',
                    "lineWidth": 1,
                }
            },
            {
                "type": 'Line',
                "data": high_rsi,
                "options": {
                    "color": 'darkred',
                    "lineWidth": 1,
                }
            },
        ]
        additional_charts.append({
            "chart": {
                "height": 150,
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
                    "rightOffset": 12,
                },
            },
            "series": rsi_series
        })

    if show_volatility:
        # Paso 3: Calcular la volatilidad.
        df = calculate_volatility(df)
        df['high_vol'] = 0.3
        df['min_vol'] = 0.1
        df['returns'] = df['returns'] * 100
        volatility = json.loads(df[['time', 'volatility']].rename(columns={"volatility": "value"}).dropna().to_json(orient="records"))
        returns = json.loads(df[['time', 'returns']].rename(columns={"returns": "value"}).dropna().to_json(orient="records"))
        high_vol = json.loads(df[['time', 'high_vol']].rename(columns={"high_vol": "value"}).dropna().to_json(orient="records"))
        min_vol = json.loads(df[['time', 'min_vol']].rename(columns={"min_vol": "value"}).dropna().to_json(orient="records"))
        volatility_series = [
            {
                "type": 'Line',
                "data": volatility,
                "options": {
                    "color": 'rgba(255, 140, 0, 0.8)',
                    "lineWidth": 1,
                }
            },
            {
                "type": 'Line',
                "data": min_vol,
                "options": {
                    "color": 'green',
                    "lineWidth": 1,
                }
            },
            {
                "type": 'Line',
                "data": high_vol,
                "options": {
                    "color": 'red',
                    "lineWidth": 1,
                }
            },
            {
                "type": 'Line',
                "data": returns,
                "priceFormat" : {
                        'type': 'price',
                        'precision' : 6,
                    'minMove':0.00001,
                                 },
                "options": {
                    "color": 'rgba(227, 177, 210, 0.4)',
                    "lineWidth": 1,
                }
            }
        ]
        additional_charts.append({
            "chart": {
                "height": 150,
                "layout": {
                    "background": {
                        "type": "solid",
                        "color": 'transparent'
                    },
                    "textColor": "white"
                },
                # "rightPriceScale":{
                #     "mode":2
                #                  },
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
                    "rightOffset": 0,
                },
            },
            "series": volatility_series
        })

    main_chart_options = {
        "height": chart_height,
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

    charts = [
        {
            "chart": main_chart_options,
            "series": price_volume_series
        }
    ]

    charts.extend(additional_charts)
    print(charts)

    return renderLightweightCharts(charts, 'overlaid')
