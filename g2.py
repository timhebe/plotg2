import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


def g2_function(x, p, q, y0, x0, a):
    return y0 + a * (1 - np.exp(-0.5 * p * np.abs(x - x0)) * (
                np.cos(0.5 * q * np.abs(x - x0)) + p / q * np.sin(0.5 * q * np.abs(x - x0))))


def plot_g2(files):
    for file in files:
        moleculeTitle = st.text_input("Molecule Title", value="")
        initial_p = st.number_input("Initial p value", value=0.4)
        initial_q = st.number_input("Initial q value", value=0.1)
        initial_y0 = st.number_input("Initial y0 value", value=500)
        initial_x0 = st.number_input("Initial x0 value", value=0)
        initial_a = st.number_input("Initial a value", value=1000)
        initial_guess = (initial_p, initial_q, initial_y0, initial_x0, initial_a)
        printInfo = st.checkbox("Print fitting info", value=False)
        xlim = st.slider("X-axis limit", -50.0, 50.0, (-50.0, 50.0))
        ylim = st.slider("Y-axis limit", 0.0, 2000.0, (0.0, 2000.0))

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

        textstr = r'$g^{(2)} (\tau = 0) = $' + str(round(g_2_0 * 100, 1)) + '%'
        max_time = max(data["Time differences (ns)"])
        max_count = max(data["Counts per bin"])

        plt.figure()
        plt.plot(data["Time differences (ns)"], data["Counts per bin"], label='Data')
        plt.plot(data["Time differences (ns)"], g2_function(data["Time differences (ns)"], *popt), linestyle='-',
                 label='Dephasing model')
        plt.title(r'$g^{(2)} (\tau)$ measurement ' + moleculeTitle)
        plt.xlabel("Time differences (ns)")
        plt.ylabel("Counts per bin")
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.text(0.05, 0.5, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)
        if xlim:
            plt.xlim(xlim)
        else:
            plt.xlim(-max_time, max_time)
        if ylim:
            plt.ylim(ylim)
        else:
            plt.ylim(0, 1.2 * max_count)
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)
