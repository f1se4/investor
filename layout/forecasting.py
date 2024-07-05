import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from calculations.calculations import (
    arima_forecasting
)
from visualizations.plotting import (
    plot_forecast_hw,
    plot_arima, plot_xgboost_forecast
)

def forecasting(data, end_time, start_time):
    # Forecasting section
    st.subheader('ForeCasting')
    if (end_time - start_time).days >= 35:
        col1, col2 = st.columns([1, 1])
        with col1:
            periods = st.number_input("Periods:", value=10, min_value=1, step=1)
        with col2:
            modelos = ["ARIMA", "Holt-Winters", "XGBoost"]
            modelo_seleccionado = st.radio("Forecasting Model:", modelos)


        if modelo_seleccionado == 'ARIMA':
            # Realizar forecasting con ARIMA
            forecast_arima = arima_forecasting(data, periods)
            arima_plot = plot_arima(data, forecast_arima, periods)
            st.pyplot(arima_plot)
            st.markdown("*ARIMA(3,1,3)*")
        elif modelo_seleccionado == 'Holt-Winters':
            # Crear el modelo Holt-Winters
            model = ExponentialSmoothing(data['Close'], trend='add', seasonal='add', seasonal_periods=12)
            fitted_model = model.fit()
            # Hacer la predicción para los próximos 5 días
            forecast = fitted_model.forecast(periods)
            st.write(f"Forecasting {periods} days for Model {modelo_seleccionado}")
            forecast_df = pd.DataFrame(forecast, columns=['forecast_values'])
            forecast_df['Date'] = pd.to_datetime(pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(forecast_df)))
            forecast_df.set_index('Date', inplace=True)
            # Mostrar gráficos de datos históricos y predicción
            forecast_plot = plot_forecast_hw(data, forecast_df)
            st.pyplot(forecast_plot)
        elif modelo_seleccionado == 'XGBoost':
            st.pyplot(plot_xgboost_forecast(data, periods))
        else:
            st.write("For an optimal forecasting you need at least 35 days")
