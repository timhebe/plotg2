import streamlit as st
# import matplotlib.pyplot as plt
import plotly.express as px
from cryo import CamReader
import io

def plot_cam(file):
    if isinstance(file, str):  # Demo mode
        name = file.split('/')[-1].split('.')[0]
        st.write(name)
    else:
        name = file.name.split('.')[0]
        st.write(name)

    CAM = CamReader(name+'.cam', True)

    """
    plt.imshow(CAM[0])
    plt.title("Cryo Measurement")
    st.pyplot(plt)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="pdf")
    st.download_button("Download as PDF", buf.getvalue(), file_name=f"{name}.pdf")
    buf.seek(0)
    plt.savefig(buf, format="png")
    st.download_button("Download as PNG", buf.getvalue(), file_name=f"{name}.png")
    """

    # new way of plotting
    labels = {'x': "X Axis Title", 'y': "X Axis Title", 'color': 'Z Label'}
    fig = px.imshow(CAM[0], labels=labels)  # aspect='equal'
    st.plotly_chart(fig)
