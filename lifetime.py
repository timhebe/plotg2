import streamlit as st
import pandas as pd
import plotly.express as px
import io

def plot_lifetime(file, device):
    with st.sidebar:
        log_scale = st.radio("Y-axis scale", ("Linear", "Log")) == "Log"
        xlim = st.slider("X-axis limit", 0.0, 50.0, (0.0, 25.0))
        ylim = st.slider("Y-axis limit", 0.0, 2000.0, (0.0, 200.0))

    if isinstance(file, str):  # Demo mode
        data = pd.read_csv(file, delimiter='\t', header=0)
    else:
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

    fig = px.line(data, x=x, y=y, labels={"y": "Counts per bin"}, title=f"Lifetime Measurement ({device})")
    if log_scale:
        fig.update_yaxes(type="log")

    fig.update_xaxes(range=[xlim[0], xlim[1]])
    fig.update_yaxes(range=[ylim[0], ylim[1]])

    st.plotly_chart(fig)

    st.download_button("Download as PDF", fig.to_image(format="pdf"), file_name=f"{file.name.split('.')[0]}.pdf")
    st.download_button("Download as PNG", fig.to_image(format="png"), file_name=f"{file.name.split('.')[0]}.png")
