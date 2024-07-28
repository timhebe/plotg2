import streamlit as st
import pandas as pd
import plotly.express as px
import io

def plot_count_rate(file, device):
    with st.sidebar:
        start_time_ps = st.number_input("Start time (ps)", value=0)
        label = st.text_input("Label", value=file.name if not isinstance(file, str) else "Example")
        plot_sum = st.radio("Plot sum of count rates?", ("No", "Yes")) == "Yes"
        title = st.text_input("Plot title", value="Count Rate vs Time")
        # xlim = st.slider("X-axis limit", 0.0, 50.0, (0.0, 25.0))
        # ylim = st.slider("Y-axis limit", 0.0, 2000.0, (0.0, 2000.0))

    if isinstance(file, str):  # Demo mode
        data = pd.read_csv(file, delimiter='\t', header=0)
    else:
        data = pd.read_csv(file, delimiter='\t', header=0)

    data["Time (s)"] = (data["Time (ps)"] - start_time_ps) / 1e12

    if plot_sum:
        data["Sum - Count rate (counts/s)"] = data["Channel 1 - Count rate (counts/s)"] + data["Channel 2 - Count rate (counts/s)"]
        fig = px.line(data, x="Time (s)", y="Sum - Count rate (counts/s)", title=title, labels={"Sum - Count rate (counts/s)": "Count rate (counts/s)"})
    else:
        fig = px.line(data, x="Time (s)", y=["Channel 1 - Count rate (counts/s)", "Channel 2 - Count rate (counts/s)"], title=title, labels={"value": "Count rate (counts/s)"})

    fig.update_xaxes(range=[xlim[0], xlim[1]])
    fig.update_yaxes(range=[ylim[0], ylim[1]])

    st.plotly_chart(fig)

    st.download_button("Download as PDF", fig.to_image(format="pdf"), file_name=f"{file.name.split('.')[0]}.pdf")
    st.download_button("Download as PNG", fig.to_image(format="png"), file_name=f"{file.name.split('.')[0]}.png")
