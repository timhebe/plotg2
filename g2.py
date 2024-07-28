import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.optimize import curve_fit
import io

def g2_function(x, p, q, y0, x0, a):
    return y0 + a * (1 - np.exp(-0.5 * p * np.abs(x-x0)) * (np.cos(0.5 * q * np.abs(x-x0)) + p/q * np.sin(0.5 * q * np.abs(x-x0))))

def plot_g2(file, device):
    with st.sidebar:
        initial_p = st.number_input("Initial guess for p", value=0.4)
        initial_q = st.number_input("Initial guess for q", value=0.1)
        initial_y0 = st.number_input("Initial guess for y0", value=500)
        initial_x0 = st.number_input("Initial guess for x0", value=0)
        initial_a = st.number_input("Initial guess for a", value=1000)
        xlim = st.slider("X-axis limit", 0.0, 50.0, (0.0, 25.0))
        ylim = st.slider("Y-axis limit", 0.0, 2000.0, (0.0, 2000.0))

    if isinstance(file, str):  # Demo mode
        data = pd.read_csv(file, delimiter='\t', header=0)
    else:
        if device == "Swabian Instruments":
            data = pd.read_csv(file, delimiter='\t', header=0)
        elif device == "PicoQuant":
            data = pd.read_csv(file, delimiter='\t', skiprows=1)
            data.columns = ['Time_ns', 'G_t']

    if device == "Swabian Instruments":
        data['Time differences (ns)'] = data['Time differences (ps)'] / 1000
        x = data["Time differences (ns)"]
        y = data["Counts per bin"]
    elif device == "PicoQuant":
        x = data["Time_ns"]
        y = data["G_t"]

    initial_guess = (initial_p, initial_q, initial_y0, initial_x0, initial_a)
    popt, _ = curve_fit(g2_function, x, y, p0=initial_guess)
    y0 = popt[2]
    a = popt[4]
    g_2_0 = g2_function(0, *popt) / (y0 + a)
    textstr = r'$g^{(2)} (\tau = 0) = $' + str(round(g_2_0 * 100, 1)) + '%'

    data['fit'] = g2_function(x, *popt)
    fig = px.line(data, x=x, y=[y, 'fit'], labels={"value": "Counts per bin"}, title=f"g2 Measurement ({device})")
    fig.add_annotation(text=textstr, xref="paper", yref="paper", x=0.05, y=0.95, showarrow=False)

    fig.update_xaxes(range=[xlim[0], xlim[1]])
    fig.update_yaxes(range=[ylim[0], ylim[1]])

    st.plotly_chart(fig)

    st.download_button("Download as PDF", fig.to_image(format="pdf"), file_name=f"{file.name.split('.')[0]}.pdf")
    st.download_button("Download as PNG", fig.to_image(format="png"), file_name=f"{file.name.split('.')[0]}.png")
