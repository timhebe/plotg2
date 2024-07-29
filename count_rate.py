import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

def plot_count_rate(file, device):
    if isinstance(file, str):  # Demo mode
        data = pd.read_csv(file, delimiter='\t', header=0)
    else:
        data = pd.read_csv(file, delimiter='\t', header=0)

    data["Time (s)"] = data["Time (ps)"] / 1e12

    plt.figure()
    plt.plot(data["Time (s)"], data["Channel 1 - Count rate (counts/s)"], label="Channel 1")
    plt.plot(data["Time (s)"], data["Channel 2 - Count rate (counts/s)"], label="Channel 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Count rate (counts/s)")
    plt.title("Count Rate vs Time")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    buf = io.BytesIO()
    plt.savefig(buf, format="pdf")
    st.download_button("Download as PDF", buf.getvalue(), file_name=f"{file.split('/')[-1].split('.')[0]}.pdf")
    buf.seek(0)
    plt.savefig(buf, format="png")
    st.download_button("Download as PNG", buf.getvalue(), file_name=f"{file.split('/')[-1].split('.')[0]}.png")
