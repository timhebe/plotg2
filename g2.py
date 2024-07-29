import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import io

def g2_function(x, p, q, y0, x0, a):
    return y0 + a * (1 - np.exp(-0.5 * p * np.abs(x-x0)) * (np.cos(0.5 * q * np.abs(x-x0)) + p/q * np.sin(0.5 * q * np.abs(x-x0))))

def plot_g2(file, device):
    with st.sidebar:
        moleculeTitle = st.text_input("Molecule Title", value="")
        initial_p = st.number_input("Initial p value", value=0.4)
        initial_q = st.number_input("Initial q value", value=0.1)
        initial_y0 = st.number_input("Initial y0 value", value=500)
        initial_x0 = st.number_input("Initial x0 value", value=0)
        initial_a = st.number_input("Initial a value", value=5e4)
        initial_guess = (initial_p, initial_q, initial_y0, initial_x0, initial_a)
        printInfo = st.checkbox("Print fitting info", value=False)

    if isinstance(file, str):  # Demo mode
        name = file.split('/')[-1].split('.')[0]
        data = pd.read_csv(file, delimiter='\t', header=0)
    else:
        name = file.name.split('.')[0]
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

    popt, _ = curve_fit(g2_function, x, y, p0=initial_guess)
    y0 = popt[2]
    a = popt[4]
    g_2_0 = g2_function(0, *popt) / (y0 + a)
    textstr = r'$g^{(2)} (\tau = 0) = $' + str(round(g_2_0 * 100, 1)) + '%'

    if printInfo:
        st.write("Optimal parameters (p, q, y0, x0, a): ", popt)
        st.write(r'$g^{(2)} (\tau = 0) = $' + str(round(g_2_0 * 100, 1)) + '%')

    plt.figure()
    plt.plot(x, y, label="Data")
    plt.plot(x, g2_function(x, *popt), label="Fit", linestyle='--')
    plt.xlabel("Time differences (ns)")
    plt.ylabel("Counts per bin")
    plt.title(f"g2 Measurement ({device})")
    plt.legend()
    plt.grid(True)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, verticalalignment='top')
    st.pyplot(plt)

    buf = io.BytesIO()
    plt.savefig(buf, format="pdf")
    st.download_button("Download as PDF", buf.getvalue(), file_name=f"{name}.pdf")
    buf.seek(0)
    plt.savefig(buf, format="png")
    st.download_button("Download as PNG", buf.getvalue(), file_name=f"{name}.png")
