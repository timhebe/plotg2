import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


def plot_count_rate(files):
    sns.set_palette("husl", len(files))
    plt.figure(figsize=(10, 6))

    start_times_ps = st.number_input(f"Start time (ps)", value=0)
    labels = st.text_input(f"Label", value="Count rate")
    plot_sum = st.radio("Plot sum of count rates?", ("No", "Yes")) == "Yes"
    title = st.text_input("Plot title", value="Count Rate vs Time")
    poster_figure = st.radio("Poster figure formatting?", ("No", "Yes")) == "Yes"
    xlim = st.slider("X-axis limit", 0.0, 50.0, (0.0, 25.0))
    ylim = st.slider("Y-axis limit", 0.0, 2000.0, (0.0, 2000.0))

    for file, start_time_ps, label in zip(files, start_times_ps, labels):
        if isinstance(file, str):  # Demo mode
            data = pd.read_csv(pd.compat.StringIO(file), delimiter='\t', header=0)
        else:
            data = pd.read_csv(file, delimiter='\t', header=0)
        data["Time (s)"] = (data["Time (ps)"] - start_time_ps) / 1e12

        if plot_sum:
            data["Sum - Count rate (counts/s)"] = data["Channel 1 - Count rate (counts/s)"] + data[
                "Channel 2 - Count rate (counts/s)"]
            plt.plot(data["Time (s)"], data["Sum - Count rate (counts/s)"], label=f'{label}')
        else:
            plt.plot(data["Time (s)"], data["Channel 1 - Count rate (counts/s)"], label=f'Channel 1 ({label})')
            plt.plot(data["Time (s)"], data["Channel 2 - Count rate (counts/s)"], label=f'Channel 2 ({label})')

    plt.xlabel('Time (s)')
    plt.ylabel('Count rate (counts/s)')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True)

    if title:
        plt.title(title)

    if poster_figure:
        formatter = FuncFormatter(lambda y, pos: f'{y / 1000:.0f}')
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlabel('Time (s)', fontweight='bold', fontsize=20, fontname='Arial')
        plt.xlim(0, 14)
        plt.ylabel('Count rate (kHz)', fontweight='bold', fontsize=20, fontname='Arial')
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(prop={'family': 'Arial', 'size': 16})
        plt.tight_layout()

    st.pyplot(plt)
