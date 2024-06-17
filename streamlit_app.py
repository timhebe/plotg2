import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import io

# Define your existing functions
def plot_count_rate(file):
    data = pd.read_csv(file, delimiter='\t', header=0)
    data["Time (s)"] = data["Time (ps)"] / 1e12

    plt.figure(figsize=(10, 6))
    plt.plot(data["Time (s)"], data["Channel 1 - Count rate (counts/s)"], label='Count rate channel 1')
    plt.plot(data["Time (s)"], data["Channel 2 - Count rate (counts/s)"], label='Count rate channel 2')

    plt.xlabel('Time (s)')
    plt.ylabel('Count rate (counts/s)')
    plt.title('Count Rate vs Time')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def hz_to_khz(y, pos):
    return f'{y / 1000:.0f}'

def plot_multiple_count_rates(files, start_times_ps, labels, plot_sum=False, title=None, poster_figure=False):
    sns.set_palette("husl", len(files))

    plt.figure(figsize=(10, 6))

    for file, start_time_ps, label in zip(files, start_times_ps, labels):
        data = pd.read_csv(file, delimiter='\t', header=0)
        data["Time (s)"] = (data["Time (ps)"] - start_time_ps) / 1e12

        if plot_sum:
            data["Sum - Count rate (counts/s)"] = data["Channel 1 - Count rate (counts/s)"] + data["Channel 2 - Count rate (counts/s)"]
            plt.plot(data["Time (s)"], data["Sum - Count rate (counts/s)"], label=f'{label}')
        else:
            plt.plot(data["Time (s)"], data["Channel 1 - Count rate (counts/s)"], label=f'Channel 1 ({label})')
            plt.plot(data["Time (s)"], data["Channel 2 - Count rate (counts/s)"], label=f'Channel 2 ({label})')

    plt.xlabel('Time (s)')
    plt.ylabel('Count rate (counts/s)')
    plt.xlim(0, 25)
    plt.legend()
    plt.grid(True)

    if title:
        plt.title(title)

    if poster_figure:
        formatter = FuncFormatter(hz_to_khz)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlabel('Time (s)', fontweight='bold', fontsize=20, fontname='Arial')
        plt.xlim(0, 14)
        plt.ylabel('Count rate (kHz)', fontweight='bold', fontsize=20, fontname='Arial')
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(prop={'family': 'Arial', 'size': 16})
        plt.tight_layout()

    st.pyplot(plt)

def g2_function(x, p, q, y0, x0, a):
    return y0 + a * (1 - np.exp(-0.5 * p * np.abs(x-x0)) * (np.cos(0.5 * q * np.abs(x-x0)) + p/q * np.sin(0.5 * q * np.abs(x-x0))))

def plot_g2(file, moleculeTitle='', initial_guess=(0.4, 0.1, 500, 0, 1000), printInfo=False):
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
    plt.plot(data["Time differences (ns)"], g2_function(data["Time differences (ns)"], *popt), linestyle='-', label='Dephasing model')
    plt.title(r'$g^{(2)} (\tau)$ measurement ' + moleculeTitle)
    plt.xlabel("Time differences (ns)")
    plt.ylabel("Counts per bin")
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.05, 0.5, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)
    plt.xlim(-max_time, max_time)
    plt.ylim(0, 1.2 * max_count)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Streamlit app code
st.title("Data Plotting Application")

uploaded_file = st.file_uploader("Choose a .txt file", type="txt", accept_multiple_files=False)
if uploaded_file is not None:
    st.write("File uploaded successfully!")
    
    # Read the first line of the file to determine the header
    file_content = uploaded_file.read().decode('utf-8')
    uploaded_file.seek(0)  # Reset file pointer to the beginning
    first_line = file_content.split('\n')[0]
    
    # Determine which function to call based on the header
    if first_line == "Time differences (ps)\tCounts per bin":
        st.write("Detected g2 data format. Plotting g2 function...")
        plot_g2(uploaded_file)
    elif first_line == "Time (ps)\tChannel 1 - Count rate (counts/s)\tChannel 2 - Count rate (counts/s)":
        st.write("Detected count rate data format. Plotting count rate...")
        plot_count_rate(uploaded_file)
    else:
        st.write("Unrecognized file format.")
