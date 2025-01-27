import streamlit as st
from PIL import Image

from calculations.calculations import (
    daily_returns, returns_vol
)
from visualizations.plotting import (
    plot_cmf_with_moving_averages,
    plot_with_indicators, plot_candlestick,
    plot_volatility, plot_ma, 
    plot_price_and_volume, plot_indicators_rsi, plot_indicators_macd
)

def analysis(data, full_data):
    # Graphic Analysis section
    st.subheader('Graphical Analysis (Selected Period)')
    selected_graph = st.radio("Graph Type", ['Line','Candle/Velas'], horizontal=True)

    st.markdown("**Overlapping Indicators**")
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        fibonacci = st.checkbox('Fibonacci')
        bollinger = st.checkbox('Bollinger Bands')

    with colp2:
        liquidity = st.checkbox('Liquidity Amihud')
        kendall = st.checkbox("Mann-Kendall")

    with colp3:
        sma200 = st.checkbox('SMA (200)')
        sma5  = st.checkbox('SMA (5)', value=True)

    markers = {
        'kendall' : kendall,
        'bollinger': bollinger,
        'liquidity' : liquidity,
        'fibonacci' : fibonacci,
        'SMA200'     : sma200,
        'SMA5'      : sma5
    }

    if selected_graph == 'Line':
        plot_full_fig = plot_price_and_volume(data, markers, full_data)
        st.plotly_chart(plot_full_fig)
    elif selected_graph == 'Candle/Velas':
        plot_candlestick_fig = plot_candlestick(data, markers, full_data)
        st.plotly_chart(plot_candlestick_fig)
        # with st.expander("Patterns"):
        #     st.image(Image.open('assets/patterns.jpg'))

    # Crear columnas
    col1, col2, col3, col4, col5 = st.columns(5)
        
    # Agregar una casilla de verificación en cada columna
    with col1:
        checkbox1 = st.checkbox('Paddle Traders')
    
    with col2:
        checkbox2 = st.checkbox('RSI')
    
    with col3:
        checkbox3 = st.checkbox('CMF & SMA')

    with col4:
        checkbox4 = st.checkbox('Volatility & Daily Returns')

    with col5:
        checkbox5 = st.checkbox('MACD')

    if checkbox1:
        # Plot synthetic indicators
        plot_ind_sintetico = plot_with_indicators(data)
        st.plotly_chart(plot_ind_sintetico)

    if checkbox2:
        plot_indicator_rsi = plot_indicators_rsi(data)
        st.plotly_chart(plot_indicator_rsi)

    if checkbox3:
        # Chaikin Money Flow & SMA
        fig_cmf_ma = plot_cmf_with_moving_averages(data)
        st.plotly_chart(fig_cmf_ma)

    if checkbox4:
        # Daily Returns & Volatility
        df_ret = daily_returns(data)
        df_vol = returns_vol(df_ret)
        #plot_vol = plot_volatility(df_vol)
        st.plotly_chart(plot_volatility(df_vol))

    if checkbox5:
        plot_indicator_macd = plot_indicators_macd(data)
        st.plotly_chart(plot_indicator_macd)

    # Smoothing plot
    st.subheader('Smoothing')

    # Crear columnas
    cols1, cols2, cols3, cols4, cols5 = st.columns(5)
        
    # Agregar una casilla de verificación en cada columna
    with cols1:
        chckbx1 = st.checkbox('Real', value=True)
        
    with cols2:
        chckbx2 = st.checkbox('Holt-Winters', value=True)
        
    with cols3:
        chckbx3 = st.checkbox('MA 20', value=True)

    with cols4:
        chckbx4 = st.checkbox('MA 50', value=True)

    with cols5:
        chckbx5 = st.checkbox('EMA 10', value=True)

    plot_ma_fig = plot_ma(data, [chckbx1, chckbx2, chckbx3, chckbx4, chckbx5])
    st.plotly_chart(plot_ma_fig)

    # bollinger = plot_bollinger_bands(data) 
    # st.plotly_chart(bollinger)
