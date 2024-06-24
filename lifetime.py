import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_lifetime(files):
    sns.set_palette("husl", len(files))
    plt.figure(figsize=(10, 6))

    labels = [st.text_input(f"Label for {file.name}", value=file.name) for file in files]
    title = st.text_input("Plot title", value="Lifetime vs Time")
    xlim = st.slider("X-axis limit", 0.0, 50.0, (0.0, 25.0))
    ylim = st.slider("Y-axis limit", 0.0, 2000.0, (0.0, 2000.0))

    for file, label in zip(files, labels):
        data = pd.read_csv(file, delimiter='\t', header=0)
        plt.plot(data["Time (s)"], data["Lifetime"], label=f'{label}')

    plt.xlabel('Time (s)')
    plt.ylabel('Lifetime')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True)

    if title:
        plt.title(title)

    st.pyplot(plt)
