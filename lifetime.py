import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy.optimize import curve_fit

# Define single exponential decay function
def exp_decay(x, y0, N0, t0, tau):
    return y0 + N0 * np.exp(-(x - t0)/tau)

# Define double exponential decay function
def double_exp_decay(x, y0, N0_1, t0_1, tau_1, N0_2, t0_2, tau_2):
    return y0 + N0_1 * np.exp(-(x - t0_1)/tau_1) + N0_2 * np.exp(-(x - t0_2)/tau_2)

def plot_lifetime(file, device):
    log_scale = st.sidebar.radio("Y-axis scale", ("Linear", "Log")) == "Log"
    fit_type = st.sidebar.selectbox("Fit Type", ["None", "Single Exponential", "Double Exponential"])

    if isinstance(file, str):  # Demo mode
        name = file.split('/')[-1].split('.')[0]
        data = pd.read_csv(file, delimiter='\t', header=0)
    else:
        name = file.name.split('.')[0]
        if device == "Swabian Instruments":
            data = pd.read_csv(file, delimiter='\t', header=0)
        elif device == "PicoQuant":
            data = pd.read_csv(file, delimiter='\t', skiprows=1)
            data.columns = ['Time_ns', 'Counts_per_bin']

    if device == "Swabian Instruments":
        data['Time (ns)'] = data['Time (ps)'] / 1000
        x = data["Time (ns)"]
        y = data["Counts per bin"]
    elif device == "PicoQuant":
        x = data["Time_ns"]
        y = data["Counts_per_bin"]

    plt.figure()
    plt.plot(x, y, label="Data")
    plt.xlabel("Time (ns)")
    plt.ylabel("Counts per bin")
    plt.title(f"Lifetime Measurement ({device})")
    if log_scale:
        plt.yscale('log')

    if fit_type == "Single Exponential":
        # Sidebar for single exponential fitting parameters
        st.sidebar.subheader("Single Exponential Fit Parameters")
        y0 = st.sidebar.slider("y0", 0.0, 1000.0, 0.0)
        N0 = st.sidebar.slider("N0", 0.0, 1000.0, max(y))
        t0 = st.sidebar.slider("t0", 0.0, max(x), 0.0)
        tau = st.sidebar.slider("tau", 0.1, max(x), 10.0)

        params = [y0, N0, t0, tau]
        popt, _ = curve_fit(exp_decay, x, y, p0=params)
        plt.plot(x, exp_decay(x, *popt), 'r--', label='Single Exp Fit: y0=%.3f, N0=%.3f, t0=%.3f, tau=%.3f' % tuple(popt))

    elif fit_type == "Double Exponential":
        # Sidebar for double exponential fitting parameters
        st.sidebar.subheader("Double Exponential Fit Parameters")
        y0 = st.sidebar.slider("y0", 0.0, 1000.0, 0.0)
        N0_1 = st.sidebar.slider("N0_1", 0.0, 1000.0, max(y))
        t0_1 = st.sidebar.slider("t0_1", 0.0, max(x), 0.0)
        tau_1 = st.sidebar.slider("tau_1", 0.1, max(x), 10.0)

        N0_2 = st.sidebar.slider("N0_2", 0.0, 1000.0, max(y)/2)
        t0_2 = st.sidebar.slider("t0_2", 0.0, max(x), 0.0)
        tau_2 = st.sidebar.slider("tau_2", 0.1, max(x), 10.0)

        params = [y0, N0_1, t0_1, tau_1, N0_2, t0_2, tau_2]
        popt, _ = curve_fit(double_exp_decay, x, y, p0=params)
        plt.plot(x, double_exp_decay(x, *popt), 'r--', label='Double Exp Fit: y0=%.3f, N0_1=%.3f, t0_1=%.3f, tau_1=%.3f, N0_2=%.3f, t0_2=%.3f, tau_2=%.3f' % tuple(popt))

    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Download buttons for the plot
    buf = io.BytesIO()
    plt.savefig(buf, format="pdf")
    st.download_button("Download as PDF", buf.getvalue(), file_name=f"{name}.pdf")
    buf.seek(0)
    plt.savefig(buf, format="png")
    st.download_button("Download as PNG", buf.getvalue(), file_name=f"{name}.png")
