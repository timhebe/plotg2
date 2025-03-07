import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import io


# Define fitting functions
def g2_grandi(x, p, q, y0, x0, a):
    return y0 + a * (1 - np.exp(-0.5 * p * np.abs(x - x0)) * (
                np.cos(0.5 * q * np.abs(x - x0)) + p / q * np.sin(0.5 * q * np.abs(x - x0))))


def g2_carmichael(x, a, y0, gamma, x0):
    return y0 + a * (1 - np.exp(-0.5 * np.abs(x - x0) * gamma)) ** 2


def g2_zirkelbach(x, a, y0, gamma, x0, S):
    return y0 + a * (1 - np.exp(-gamma * (1 + S) * np.abs(x - x0)))


def read_data(file, device):
    if device == "Swabian Instruments":
        data = pd.read_csv(file, delimiter='\t', header=0)
    elif device == "PicoQuant":
        data = pd.read_csv(file, delimiter='\t', skiprows=1)
        data.columns = ['Time[ns]', 'G(t)[]']
    return data


def fit_g2(x, y, model, initial_guess):
    fit_functions = {
        "Grandi et al.": (g2_grandi, ["p", "q", "y0", "x0", "a"]),
        "Carmichael et al.": (g2_carmichael, ["a", "y0", "gamma", "x0"]),
        "Zirkelbach et al.": (g2_zirkelbach, ["a", "y0", "gamma", "x0", "S"]),
    }
    func, param_names = fit_functions[model]
    params, cov = curve_fit(func, x, y, p0=initial_guess)
    errors = np.sqrt(np.diag(cov))
    residuals = y - func(x, *params)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    chi_sq_red = ss_res / (len(y) - len(params))
    return params, errors, r_squared, chi_sq_red, param_names


def plot_g2(file, device):
    with st.sidebar:
        model = st.selectbox("Choose Fit Model", ["Grandi et al.", "Carmichael et al.", "Zirkelbach et al."])
        initial_x0 = st.number_input("Initial x0", value=0.0)
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
    initial_guesses = {
        "Grandi et al.": [0.4, 0.1, initial_y0, initial_x0, initial_a],
        "Carmichael et al.": [initial_a, initial_y0, 0.1, initial_x0],
        "Zirkelbach et al.": [initial_a, initial_y0, 0.1, initial_x0, 1]
    }
    params, errors, r_squared, chi_sq_red, param_names = fit_g2(x, y, model, initial_guesses[model])
    if model == "Grandi et al.":
        g2_0 = g2_grandi(params[3], *params[:5]) / (params[2] + params[-1])
    elif model == "Carmichael et al.":
        g2_0 = g2_carmichael(params[3], *params) / (params[1] + params[0])
    elif model == "Zirkelbach et al.":
        g2_0 = g2_zirkelbach(params[3], *params) / (params[1] + params[0])
    textstr = rf'$g^{{(2)}} (\tau = 0) = {round(g2_0 * 100, 1)}\%$'

    if printInfo:
        st.markdown("**More Information:**")
        links = {
            "Grandi et al.": "[Grandi et al. Publication](https://doi.org/10.1103/PhysRevA.94.063839)",
            "Carmichael et al.": "[Carmichael et al. Publication](https://doi.org/10.1088/0022-3700/9/8/007)",
            "Zirkelbach et al.": "[Zirkelbach et al. Publication](https://doi.org/10.1103/PhysRevLett.125.103603)"
        }
        st.markdown(links[model])
        formulas = {
            "Grandi et al.": r"$g^{(2)}(\tau) = 1 - e^{-\frac{p}{2} \tau} \left[ \cos \left( \frac{q}{2} \tau \right) + \frac{p}{q} \sin \left( \frac{q}{2} \tau \right) \right]$",
            "Carmichael et al.": r"$g^{(2)}(\tau) = \left( 1 - e^{-\frac{1}{2} \Gamma_1 \tau} \right)^2$",
            "Zirkelbach et al.": r"$g^{(2)}_{01}(\tau) = 1 - e^{- \Gamma_1 (1+S) |\tau|}$"
        }
        st.write(formulas[model])

        st.write(f"**{model} Fit Report**")
        for name, value, error in zip(param_names, params, errors):
            st.write(f"{name} = {value:.6f} ± {error:.6f}")
        st.write(f"R² = {r_squared:.6f}, Reduced χ² = {chi_sq_red:.6f}")

    plt.figure()
    plt.plot(x, y, label="Data")
    plt.plot(x, eval(f"g2_{model.split()[0].lower()}(x, *params)"), label="Fit", linestyle='--')
    plt.xlabel("Time differences (ns)")
    plt.ylabel("Counts per bin" if device == "Swabian Instruments" else r"$g^{(2)} (\tau)$")
    plt.title(r"$g^{(2)}$ " + f"Measurement ({device})")
    plt.ylim(0, None)
    plt.legend()
    plt.grid(True)
    plt.text(0.05, 0.25, textstr, transform=plt.gca().transAxes, verticalalignment='top')
    st.pyplot(plt)

    buf = io.BytesIO()
    plt.savefig(buf, format="pdf")
    if file == "example_data/example g2 Swabian.txt" or file == "example_data/example g2 PicoQuant.dat":
        st.download_button("Download as PDF", buf.getvalue(), file_name="example g2.pdf")
    else:
        st.download_button("Download as PDF", buf.getvalue(), file_name=f"{file.name.split('.')[0]}.pdf")
    buf.seek(0)
    plt.savefig(buf, format="png")
    if file == "example_data/example g2 Swabian.txt" or file == "example_data/example g2 PicoQuant.dat":
        st.download_button("Download as PNG", buf.getvalue(), file_name="example g2.png")
    else:
        st.download_button("Download as PNG", buf.getvalue(), file_name=f"{file.name.split('.')[0]}.png")
