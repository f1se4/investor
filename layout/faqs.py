import streamlit as st

def faqs():
    st.markdown("""
            **Volumen**:

            Este gráfico muestra los retornos diarios logarítmicos del precio de cierre del activo y la volatilidad asociada.
            - **Retornos Diarios:** Representan el cambio porcentual en el precio de cierre de un día a otro.
            - **Volatilidad:** Es la desviación estándar de los retornos diarios móviles (últimos 12 días), que indica la variabilidad en los cambios de precio.
            - La volatilidad alta puede indicar fluctuaciones significativas en el precio del activo.
            """)
    st.markdown('''
            **Chaikin Money**:

            Flow (CMF) es un indicador técnico utilizado en el análisis financiero para medir la acumulación o distribución de un activo, basado en el volumen y el precio.

            El CMF puede ayudar a confirmar si la tendencia observada en las medias móviles está respaldada por el volumen. Por ejemplo, una media móvil que indica una tendencia alcista acompañada por un CMF positivo sugiere que la tendencia está respaldada por una presión de compra sostenida.

            - **Identificación de Divergencias:** Comparar el CMF con medias móviles puede ayudar a identificar divergencias entre el precio y el volumen. Si las medias móviles indican una tendencia alcista pero el CMF está en negativo, podría indicar que, a pesar de la subida del precio, hay una acumulación de presión de venta, lo que podría señalar una posible reversión de la tendencia.

            - **Validez de Señales de Compra/Venta:** Las medias móviles se utilizan a menudo para generar señales de compra y venta cuando cruzan ciertos niveles. El CMF puede validar estas señales. Por ejemplo, una señal de compra basada en un cruce de media móvil es más fiable si el CMF es positivo, ya que indica que la presión de compra está respaldando la subida del precio. Análisis de Sentimiento del Mercado:

            Las medias móviles reflejan la tendencia del precio a lo largo del tiempo. El CMF, al considerar el volumen, proporciona información sobre el sentimiento del mercado (acumulación o distribución). Comparar ambos puede ofrecer una visión más completa de la situación del mercado.
               ''')
    st.markdown("""
                - **RSI (Relative Strength Index):**
                  - El RSI es un indicador de momentum que mide la velocidad y el cambio de los movimientos de precios.
                  - Se calcula como un valor oscilante entre 0 y 120 y se utiliza comúnmente para identificar condiciones de sobrecompra (por encima de 70) y sobreventa (por debajo de 30).
                  - Cuando el RSI está por encima de 70, el activo se considera sobrecomprado, lo que podría indicar una posible corrección a la baja.
                  - Por el contrario, cuando el RSI está por debajo de 30, el activo se considera sobrevendido, lo que podría señalar una oportunidad de compra.
                """)
    st.markdown("""
                - **MACD (Moving Average Convergence Divergence):**
                    - El MACD es un indicador de seguimiento de tendencias que muestra la diferencia entre dos medias móviles exponenciales (EMA) de distintos períodos.
                    - El MACD Line se representa como una línea sólida y la Signal Line como una línea punteada. Cuando la Línea MACD está por encima de la Línea de Señal, a menudo se colorea de verde.
                    - Esto suele indicar un **impulso alcista** y es considerado un signo positivo para el activo, sugiriendo una tendencia alcista en el corto plazo.
                    - Histograma del MACD: Además de las líneas, el MACD también se representa mediante un histograma que muestra la diferencia entre la Línea MACD y la Signal Line. En muchos gráficos, las barras del histograma se pintan de verde cuando la Línea MACD está por encima de la Signal Line, lo que refuerza la indicación de una tendencia alcista.
                """)
