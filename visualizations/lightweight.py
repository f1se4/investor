from streamlit_lightweight_charts import renderLightweightCharts
import numpy as np
import json

COLOR_BULL = 'rgba(38,166,154,0.9)' # #26a69a
COLOR_BEAR = 'rgba(239,83,80,0.9)'  # #ef5350

def calculate_sma(df, window):
    return df['close'].rolling(window=window).mean()

def calculate_macd(df):
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def f_daily_plot(df, df_sm, show_sma200=False, show_sma5=False, show_macd=False, show_rsi=False, show_volatility=False, chart_height=500):
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df_sm.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df['time'] = df['time'].view('int64') // 10**9
    df_sm['time'] = df_sm['time'].view('int64') // 10**9

    df['color'] = np.where(df['open'] > df['close'], COLOR_BEAR, COLOR_BULL)  # bull or bear

    candles = json.loads(df.to_json(orient="records"))
    volume = json.loads(df[['time', 'volume']].rename(columns={"volume": "value"}).to_json(orient="records"))
    
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
        max_time = df['time'].max()
        min_time = df['time'].min()

        df_sm = df_sm[(df_sm['time'] >= min_time) & (df_sm['time'] <= max_time)]

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

    additional_charts = []

    if show_macd:
        df['macd'], df['signal'] = calculate_macd(df)
        macd = json.loads(df[['time', 'macd']].rename(columns={"macd": "value"}).dropna().to_json(orient="records"))
        signal = json.loads(df[['time', 'signal']].rename(columns={"signal": "value"}).dropna().to_json(orient="records"))
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
        rsi_series = [
            {
                "type": 'Line',
                "data": rsi,
                "options": {
                    "color": 'rgba(0, 0, 255, 0.8)',
                    "lineWidth": 1,
                }
            }
        ]
        additional_charts.append({
            "chart": {
                "height": 100,
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
        df['volatility'] = df['high'] - df['low']
        volatility = json.loads(df[['time', 'volatility']].rename(columns={"volatility": "value"}).dropna().to_json(orient="records"))
        volatility_series = [
            {
                "type": 'Line',
                "data": volatility,
                "options": {
                    "color": 'rgba(255, 140, 0, 0.8)',
                    "lineWidth": 1,
                }
            }
        ]
        additional_charts.append({
            "chart": {
                "height": 100,
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

    return renderLightweightCharts(charts, 'priceAndVolume')
