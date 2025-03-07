import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import io


# https://doi.org/10.1103/PhysRevA.94.063839
def function_grandi(x, p, q, y0, x0, a):
    return y0 + a * (1 - np.exp(-0.5 * p * np.abs(x - x0)) * (
                np.cos(0.5 * q * np.abs(x - x0)) + p / q * np.sin(0.5 * q * np.abs(x - x0))))


# https://doi.org/10.1088/0022-3700/9/8/007
def function_carmichael(x, a, y0, gamma, x0):
    return y0 + a * (1 - np.exp(-0.5 * np.abs(x - x0) * gamma)) ** 2


# https://doi.org/10.1103/PhysRevLett.125.103603
def function_zirkelbach(x, a, y0, gamma, x0, S):
    return y0 + a * (1 - np.exp(-gamma * (1 + S) * np.abs(x - x0)))


def read_data(file, device):
    if device == "Swabian Instruments":
        data = pd.read_csv(file, delimiter='\t', header=0)
    elif device == "PicoQuant":
        data = pd.read_csv(file, delimiter='\t', skiprows=1)
        data.columns = ['Time[ns]', 'G(t)[]']
    return data


def fit_g2(x, y, model, initial_guess):
    if model == "Grandi et al.":
        fit_function = function_grandi
    elif model == "Carmichael et al.":
        fit_function = function_carmichael
    elif model == "Zirkelbach et al.":
        fit_function = function_zirkelbach
    else:
        raise ValueError("Invalid model selected.")

    popt, _ = curve_fit(fit_function, x, y, p0=initial_guess)
    return popt


def plot_g2(file, device):
    global g2_t0, initial_guess, y, x
    with st.sidebar:
        model = st.selectbox("Select Fit Model", ["Grandi et al.", "Carmichael et al.", "Zirkelbach et al."])
        initial_x0 = st.number_input("Initial x0 value", value=0)
        printInfo = st.checkbox("Print fitting info", value=False)

    data = read_data(file, device)
    if device == "Swabian Instruments":
        data['Time differences (ns)'] = data['Time differences (ps)'] / 1000
        x = data["Time differences (ns)"]
        y = data["Counts per bin"]
    elif device == "PicoQuant":
        x = data["Time[ns]"]
        y = data["G(t)[]"]

    initial_y0 = min(y)
    initial_a = max(y) - initial_y0

    if model == "Grandi et al.":
        initial_guess = (0.4, 0.1, initial_y0, initial_x0, initial_a)
    elif model == "Carmichael et al.":
        initial_guess = (initial_a, initial_y0, 0.1, initial_x0)
    elif model == "Zirkelbach et al.":
        initial_guess = (initial_a, initial_y0, 0.1, initial_x0, 1)

    popt = fit_g2(x, y, model, initial_guess)

    if model == "Grandi et al.":
        x0 = popt[3]
        y0 = popt[2]
        a = popt[4]
        g2_t0 = function_grandi(x0, *popt) / (y0 + a)
    elif model == "Carmichael et al.":
        x0 = popt[3]
        y0 = popt[1]
        a = popt[0]
        g2_t0 = function_carmichael(x0, *popt) / (y0 + a)
    elif model == "Zirkelbach et al.":
        x0 = popt[3]
        y0 = popt[1]
        a = popt[0]
        g2_t0 = function_zirkelbach(x0, *popt) / (y0 + a)

    textstr = rf'$g^{{(2)}} (\tau = 0) = {round(g2_t0 * 100, 1)}\%$'

    if printInfo:
        st.write(f"Optimal parameters for {model}: ", popt)
        st.write(textstr)

    plt.figure()
    plt.plot(x, y, label="Data")
    if model == "Grandi et al.":
        plt.plot(x, function_grandi(x, *popt), label=f"Fit ({model})", linestyle='--')
    elif model == "Carmichael et al.":
        plt.plot(x, function_carmichael(x, *popt), label=f"Fit ({model})", linestyle='--')
    elif model == "Zirkelbach et al.":
        plt.plot(x, function_zirkelbach(x, *popt), label=f"Fit ({model})", linestyle='--')
    plt.xlabel(r"Time $\tau$ (ns)")
    plt.ylabel("Counts per bin" if device == "Swabian Instruments" else r"$g^{(2)} (\tau)$")
    plt.title(rf"$g^[(2)]$ Measurement ({device})")
    plt.ylim(0, None)
    plt.legend()
    plt.grid(True)
    plt.text(0.05, 0.25, textstr, transform=plt.gca().transAxes, verticalalignment='top')
    st.pyplot(plt)

    buf = io.BytesIO()
    plt.savefig(buf, format="pdf")
    st.download_button("Download as PDF", buf.getvalue(), file_name=f"{file.name.split('.')[0]}.pdf")
    buf.seek(0)
    plt.savefig(buf, format="png")
    st.download_button("Download as PNG", buf.getvalue(), file_name=f"{file.name.split('.')[0]}.png")