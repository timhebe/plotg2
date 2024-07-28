import streamlit as st
import matplotlib.pyplot as plt
from cryo import CamReader
import io

def plot_cam(file):
    with st.sidebar:
        title = st.text_input("Plot title", value="Cryo Measurement")

    PSF = CamReader(file.name, True)
    plt.imshow(PSF[0])
    plt.title(title)
    st.pyplot()

    buf = io.BytesIO()
    plt.savefig(buf, format="pdf")
    st.download_button("Download as PDF", buf.getvalue(), file_name=f"{file.name.split('.')[0]}.pdf")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    st.download_button("Download as PNG", buf.getvalue(), file_name=f"{file.name.split('.')[0]}.png")
