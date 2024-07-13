import yfinance as yf

from visualizations.lightweight import f_daily_plot

def daily(ticker_data, selected_interval):
    # Historical data
    daily_data = yf.download(ticker_data.get_info()['symbol'],period='5d',interval=selected_interval).drop(columns=['Adj Close'])

    # Convert index (date) to CEST
#    try:
    daily_data.index = daily_data.index.tz_convert('CET')
#    except:
#        pass

    daily_data = daily_data.reset_index()
    print(daily_data)
    
    f_daily_plot(daily_data)
