import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

def read_data(file, device):
    global data
    if device == "Princeton Instruments":
        data = pd.read_csv(file)

    elif device == "Andor, Oxford Instruments":
        wavelengths = []
        intensities = []
        # Read the file manually line by line
        with open(file, 'r') as dataset:
            for line in dataset:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    wavelengths.append(float(parts[0]))
                    intensities.append(float(parts[1]))

        # Create a DataFrame from the lists
        data = pd.DataFrame({
            'Wavelength': wavelengths,
            'Intensity': intensities
        })

    return data

def plot_spectrum(file, device):
    if isinstance(file, str):  # Demo mode
        name = file.split('/')[-1].split('.')[0]
        data = read_data(file, device)
    else:
        name = file.name.split('.')[0]
        data = read_data(file, device)

    plt.figure()
    plt.plot(data['Wavelength'], data['Intensity'], label='Spectrum')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.title(f'Spectrum ({device})')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    buf = io.BytesIO()
    plt.savefig(buf, format="pdf")
    st.download_button("Download as PDF", buf.getvalue(), file_name=f"{name}.pdf")
    buf.seek(0)
    plt.savefig(buf, format="png")
    st.download_button("Download as PNG", buf.getvalue(), file_name=f"{name}.png")
