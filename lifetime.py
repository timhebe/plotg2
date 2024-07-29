import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

def plot_lifetime(file, device):
    log_scale = st.sidebar.radio("Y-axis scale", ("Linear", "Log")) == "Log"

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
    plt.grid(True)
    st.pyplot(plt)

    buf = io.BytesIO()
    plt.savefig(buf, format="pdf")
    st.download_button("Download as PDF", buf.getvalue(), file_name=f"{name}.pdf")
    buf.seek(0)
    plt.savefig(buf, format="png")
    st.download_button("Download as PNG", buf.getvalue(), file_name=f"{name}.png")
