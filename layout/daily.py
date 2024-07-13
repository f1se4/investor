import yfinance as yf
from visualizations.lightweight import f_daily_plot

def daily(data, data_sm, show_sma200, show_sma5, show_macd, show_rsi, show_volatility):
    # Convert index (date) to CEST
    #data.index = data.index.tz_convert('CET')
    data = data.reset_index()
    data_sm = data_sm.reset_index()
    
    f_daily_plot(data, data_sm, show_sma200, show_sma5, show_macd, show_rsi, show_volatility)
