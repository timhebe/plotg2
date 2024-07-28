import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

def plot_spectrum(file, device):
    if isinstance(file, str):  # Demo mode
        data = pd.read_csv(file, delimiter='\t', header=0)
    else:
        if device == "Princeton Instruments":
            data = pd.read_csv(file, delimiter='\t', header=0)
        elif device == "Andor, Oxford Instruments":
            data = pd.read_csv(file, delimiter=',', header=None)
            data.columns = ['Wavelength', 'Intensity']

    plt.figure()
    plt.plot(data['Wavelength'], data['Intensity'], label='Spectrum')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.title(f'Spectrum ({device})')
    plt.legend()
    plt.grid(True)
    st.pyplot()

    buf = io.BytesIO()
    plt.savefig(buf, format="pdf")
    st.download_button("Download as PDF", buf.getvalue(), file_name=f"{file.split('/')[-1].split('.')[0]}.pdf")
    buf.seek(0)
    plt.savefig(buf, format="png")
    st.download_button("Download as PNG", buf.getvalue(), file_name=f"{file.split('/')[-1].split('.')[0]}.png")
