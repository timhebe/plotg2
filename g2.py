import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.optimize import curve_fit
import numpy as np
import io


def g2_function(x, p, q, y0, x0, a):
    return y0 + a * (1 - np.exp(-0.5 * p * np.abs(x - x0)) * (
                np.cos(0.5 * q * np.abs(x - x0)) + p / q * np.sin(0.5 * q * np.abs(x - x0))))


def plot_g2(file):
    with st.sidebar:
        moleculeTitle = st.text_input("Molecule Title", value="")
        initial_p = st.number_input("Initial p value", value=0.4)
        initial_q = st.number_input("Initial q value", value=0.1)
        initial_y0 = st.number_input("Initial y0 value", value=500)
        initial_x0 = st.number_input("Initial x0 value", value=0)
        initial_a = st.number_input("Initial a value", value=5e4)
        initial_guess = (initial_p, initial_q, initial_y0, initial_x0, initial_a)
        printInfo = st.checkbox("Print fitting info", value=False)
        # xlim = st.slider("X-axis limit", -50.0, 50.0, (-50.0, 50.0))
        # ylim = st.slider("Y-axis limit", 0.0, 2000.0, (0.0, 2000.0))

    if isinstance(file, str):  # Demo mode
        data = pd.read_csv(io.StringIO(file), sep='\t')
    else:
        data = pd.read_csv(file, sep='\t')
    data['Time differences (ns)'] = data['Time differences (ps)'] / 1000
    popt, _ = curve_fit(g2_function, data["Time differences (ns)"], data["Counts per bin"], p0=initial_guess)

    if printInfo:
        st.write("Optimal parameters (p, q, y0, x0, a): ", popt)

    y0 = popt[2]
    a = popt[4]
    g_2_0 = g2_function(0, *popt) / (y0 + a)

    if printInfo:
        st.write("g^2 (0) = ", round(g_2_0 * 100, 1), "%")

    data['Fitted'] = g2_function(data["Time differences (ns)"], *popt)

    fig = px.line(data, x="Time differences (ns)", y=["Counts per bin", "Fitted"],
                  title=f'g^2 (0) Measurement {moleculeTitle}')
    fig.update_layout()  # xaxis_range=xlim, yaxis_range=ylim)
    st.plotly_chart(fig)
