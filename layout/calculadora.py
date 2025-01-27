import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from calculations.calculations import calcular_impuesto

def calculadora(initial_investment, monthly_contribution, annual_interest_rate, years):

    # Calculating values
    months = years * 12
    monthly_interest_rate = annual_interest_rate / 100 / 12

    # Lists to store results
    balances = []
    initial_contributions = []
    total_contributions = []
    interests_generated = []

    # Initial balance is the initial investment
    balance = initial_investment
    total_invested = initial_investment

    for month in range(1, months + 1):
        interest = balance * monthly_interest_rate
        balance += interest + monthly_contribution

        # Update total invested
        total_invested += monthly_contribution

        # Track data for each month
        balances.append(balance)
        initial_contributions.append(initial_investment if month == 1 else 0)
        total_contributions.append(total_invested)
        interests_generated.append(balance - total_invested)

    # Summary
    total_invested_amount = initial_investment + monthly_contribution * months
    total_balance = balances[-1]
    total_gain = total_balance - total_invested_amount

    impuesto = calcular_impuesto(total_gain) 
    beneficio = total_gain - impuesto

    st.title("Compound Interest Calculator")
    st.header("Summary")
    st.markdown(f"**Total Gain**: <span style='color: darkgreen;'>{total_gain:,.2f}€</span>", unsafe_allow_html=True)
    st.markdown(f"**Total Investment**: <span style='color: orange;'>{total_invested_amount:,.2f}€</span>", unsafe_allow_html=True)
    st.markdown(f"**Taxes**: <span style='color: red;'>{impuesto:,.2f}€</span>", unsafe_allow_html=True)
    st.markdown(f"**Total Gain - Taxes**: <span style='color: green;'>{beneficio:,.2f}€</span>", unsafe_allow_html=True)
    st.markdown(f"**Final Capital**: <span style='color: #b4ac85;'>{total_balance - impuesto:,.2f}€</span>", unsafe_allow_html=True)

    # Create a DataFrame for the chart
    df = pd.DataFrame({
        "Month": np.arange(1, months + 1),
        "Total Balance (€)": balances,
        "Initial Investment (€)": np.cumsum(initial_contributions),
        "Monthly Contributions (€)": monthly_contribution * np.arange(1, months + 1),
        "Generated Interest (€)": interests_generated
    })

    # Interactive chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Month"],
        y=df["Total Balance (€)"],
        mode='lines',
        name='Total Balance (€)',
        line=dict(color='#b4ac85', width=2)
    ))

    fig.add_trace(go.Bar(
        x=df["Month"],
        y=df["Initial Investment (€)"],
        name='Initial Investment (€)',
        marker=dict(color='#83a7ae')
    ))

    fig.add_trace(go.Bar(
        x=df["Month"],
        y=df["Monthly Contributions (€)"],
        name='Monthly Contributions (€)',
        marker=dict(color='#206b6c')
    ))

    fig.add_trace(go.Bar(
        x=df["Month"],
        y=df["Generated Interest (€)"],
        name='Generated Interest (€)',
        marker=dict(color='#305d7e')
    ))

    fig.update_layout(
        barmode='stack',
        #title='Investment Growth with Compound Interest',
        xaxis_title='Month',
        yaxis_title='Amount (€)',
        showlegend=False,
        legend_title='Components'
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    # Display the data table
    st.subheader("Detailed Data")
    st.dataframe(df.set_index('Month'))

