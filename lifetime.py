import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# Define single exponential decay function
def exp_decay(x, y0, N0, t0, tau):
    return y0 + N0 * np.exp(-(x - t0) / tau)


# Define double exponential decay function
def double_exp_decay(x, y0, N0_1, t0_1, tau_1, N0_2, t0_2, tau_2):
    return y0 + N0_1 * np.exp(-(x - t0_1) / tau_1) + N0_2 * np.exp(-(x - t0_2) / tau_2)


def plot_lifetime(file, device):
    log_scale = st.sidebar.radio("### Y-axis scale", ("Linear", "Log")) == "Log"
    fit_type = st.sidebar.selectbox("### Fit Type", ["None", "Single Exponential", "Double Exponential"])
    # show_fit_params = st.sidebar.checkbox("Show Fit Parameters", value=False)

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

    # X-axis limits
    st.sidebar.markdown("### X-axis limits")
    x_min_default = float(min(x))
    x_max_default = float(max(x))
    x_min = st.sidebar.number_input('X min (ns)', value=x_min_default)
    x_max = st.sidebar.number_input('X max (ns)', value=x_max_default)

    plt.figure()
    plt.plot(x, y, label="Data")
    plt.xlabel("Time (ns)")
    plt.ylabel("Counts per bin")
    plt.title(f"Lifetime Measurement ({device})")
    plt.xlim(x_min, x_max)
    if log_scale:
        plt.yscale('log')

    if fit_type != "None":
        # Peak finding
        peaks, properties = find_peaks(y, prominence=np.max(y) / 2)
        x_pk = x[peaks]
        data_pk = y[peaks]

        to_delete = np.where(data_pk < max(data_pk) / 100)
        data_pk = np.delete(data_pk, to_delete)
        x_pk = np.delete(x_pk, to_delete)

        first_peak = x_pk[0]  # in ns
        peak_height = data_pk[0]  # counts at first peak

        # Start and stop in [ns]
        # We will fit between start and stop only. These parameters influence the fit a lot.
        # Start at the first peak + 1 ns, stop at a sensible default based on data range.
        x_data_max = float(max(x))
        default_start = min(first_peak + 1, x_data_max - 1)
        default_stop = min(first_peak + 151, x_data_max)

        st.sidebar.markdown("### Fit range")
        start = st.sidebar.number_input('Start (in ns)', 0.0, x_data_max, default_start)
        stop = st.sidebar.number_input('Stop (in ns)', 0.0, x_data_max, default_stop)
        # Convert start and stop times (in ns) to indices
        start_idx = np.searchsorted(x, start)
        stop_idx = np.searchsorted(x, stop)
        data_fit = y[start_idx:stop_idx]
        X_fit = np.linspace(0, stop - start, num=len(data_fit))

        plt.plot(x_pk, data_pk, 'o', label="Peaks")
        plt.axvline(start, linestyle="--", color="seagreen")
        plt.axvline(stop, linestyle="--", color="firebrick")

        # Parameter name to LaTeX mapping
        param_latex = {
            'y0': r'y_0',
            'N0': r'N_0',
            't0': r't_0',
            'tau': r'\tau',
            'N0_1': r'N_{0,1}',
            't0_1': r't_{0,1}',
            'tau_1': r'\tau_1',
            'N0_2': r'N_{0,2}',
            't0_2': r't_{0,2}',
            'tau_2': r'\tau_2',
        }

        if fit_type == "Single Exponential":
            y0, N0, t0, tau = [np.min(data_fit), peak_height, 0, 10]
            params = [y0, N0, t0, tau]
            param_names = ['y0', 'N0', 't0', 'tau']
            popt, pcov = curve_fit(exp_decay, X_fit, data_fit, p0=params, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            Y_fit = exp_decay(X_fit, *popt)
            formula = r"y(x) = y_0 + N_0 \cdot \exp\left(\frac{-(x - t_0)}{\tau}\right)"

            """
            # Sidebar for single exponential fitting parameters
            if show_fit_params:
                st.sidebar.subheader("Single Exponential Fit Parameters")
                y0 = st.sidebar.slider("y0", 0.0, 1000.0, 0.0)
                N0 = st.sidebar.slider("N0", 0.0, 1000.0, float(max(y)))
                t0 = st.sidebar.slider("t0", 0.0, float(max(x)), 0.0)
                tau = st.sidebar.slider("tau", 0.1, float(max(x)), 10.0)
                # to be finished...
            """

        elif fit_type == "Double Exponential":
            y0, N0_1, t0_1, tau_1, N0_2, t0_2, tau_2, = [np.min(data_fit), peak_height, 0, 10, peak_height / 2, 0, 10]
            params = [y0, N0_1, t0_1, tau_1, N0_2, t0_2, tau_2]
            param_names = ['y0', 'N0_1', 't0_1', 'tau_1', 'N0_2', 't0_2', 'tau_2']
            popt, pcov = curve_fit(double_exp_decay, X_fit, data_fit, p0=params, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            Y_fit = double_exp_decay(X_fit, *popt)
            formula = r"y(x) = y_0 + N_{0,1} \cdot \exp\left(\frac{-(x - t_{0,1})}{\tau_1}\right) + N_{0,2} \cdot \exp\left(\frac{-(x - t_{0,2})}{\tau_2}\right)"

            """
            # Sidebar for double exponential fitting parameters
            if show_fit_params:
                st.sidebar.subheader("Double Exponential Fit Parameters")
                y0 = st.sidebar.slider("y0", 0.0, 1000.0, 0.0)
                N0_1 = st.sidebar.slider("N0_1", 0.0, 1000.0, float(max(y)))
                t0_1 = st.sidebar.slider("t0_1", 0.0, float(max(x)), 0.0)
                tau_1 = st.sidebar.slider("tau_1", 0.1, float(max(x)), 10.0)

                N0_2 = st.sidebar.slider("N0_2", 0.0, 1000.0, float(max(y)) / 2)
                t0_2 = st.sidebar.slider("t0_2", 0.0, float(max(x)), 0.0)
                tau_2 = st.sidebar.slider("tau_2", 0.1, float(max(x)), 10.0)
            """

        plt.plot(X_fit + start, Y_fit, 'r--', label=f'{fit_type} Fit')

        # Always show fit results
        st.markdown("### Fitting Formula")
        st.latex(formula)
        st.markdown("### Fitting Parameters")
        for name, param, err in zip(param_names, popt, perr):
            latex_name = param_latex.get(name, name)
            st.latex(rf"{latex_name} = {param:.3f} \pm {err:.3f}")

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