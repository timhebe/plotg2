import streamlit as st
import pandas as pd
import plotly.express as px
import io


def plot_count_rate(file):
    with st.sidebar:
        start_time_ps = st.number_input("Start time (ps)", value=0)
        label = st.text_input("Label", value=file.name if not isinstance(file, str) else "Example")
        plot_sum = st.radio("Plot sum of count rates?", ("No", "Yes")) == "Yes"
        title = st.text_input("Plot title", value="Count Rate vs Time")
        # xlim = st.slider("X-axis limit", 0.0, 50.0, (0.0, 25.0))
        # ylim = st.slider("Y-axis limit", 0.0, 2000.0, (0.0, 2000.0))

    if isinstance(file, str):  # Demo mode
        data = pd.read_csv(io.StringIO(file), delimiter='\t', header=0)
    else:
        data = pd.read_csv(file, delimiter='\t', header=0)

    data["Time (s)"] = (data["Time (ps)"] - start_time_ps) / 1e12

    if plot_sum:
        data["Sum - Count rate (counts/s)"] = data["Channel 1 - Count rate (counts/s)"] + data[
            "Channel 2 - Count rate (counts/s)"]
        fig = px.line(data, x="Time (s)", y="Sum - Count rate (counts/s)", title=title, labels={'value': label})
    else:
        fig = px.line(data, x="Time (s)", y=["Channel 1 - Count rate (counts/s)", "Channel 2 - Count rate (counts/s)"],
                      title=title, labels={'value': label})

    fig.update_layout()  # xaxis_range=xlim, yaxis_range=ylim)
    st.plotly_chart(fig)
