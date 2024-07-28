import streamlit as st
import pandas as pd
import plotly.express as px
import io

def plot_spectra(file, device):
    with st.sidebar:
        xlim = st.slider("X-axis limit", 400.0, 800.0, (400.0, 700.0))
        ylim = st.slider("Y-axis limit", 0.0, 2000.0, (0.0, 1000.0))

    if isinstance(file, str):  # Demo mode
        data = pd.read_csv(file, delimiter='\t', header=0)
    else:
        if device == "Princeton Instruments":
            data = pd.read_csv(file, delimiter='\t', header=0)
        elif device == "Andor, Oxford Instruments":
            data = pd.read_csv(file, delimiter=',', header=None)
            data.columns = ['Wavelength', 'Intensity']

    fig = px.line(data, x="Wavelength", y="Intensity", title=f"Spectrum ({device})")
    fig.update_xaxes(range=[xlim[0], xlim[1]])
    fig.update_yaxes(range=[ylim[0], ylim[1]])

    st.plotly_chart(fig)

    st.download_button("Download as PDF", fig.to_image(format="pdf"), file_name=f"{file.name.split('.')[0]}.pdf")
    st.download_button("Download as PNG", fig.to_image(format="png"), file_name=f"{file.name.split('.')[0]}.png")
