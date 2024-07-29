import streamlit as st
import matplotlib.pyplot as plt
from cryo import CamReader
import io

def plot_cam(file):
    if isinstance(file, str):  # Demo mode
        name = file.split('/')[-1].split('.')[0]
        st.write(name)
    else:
        name = file.name.split('.')[0]
        st.write(name)

    PSF = CamReader(name+'.cam', True)
    plt.imshow(PSF[0])
    plt.title("Cryo Measurement")
    st.pyplot()

    buf = io.BytesIO()
    plt.savefig(buf, format="pdf")
    st.download_button("Download as PDF", buf.getvalue(), file_name=f"{name}.pdf")
    buf.seek(0)
    plt.savefig(buf, format="png")
    st.download_button("Download as PNG", buf.getvalue(), file_name=f"{name}.png")
