import streamlit as st
import pandas as pd
import plotly.express as px

def plot_lifetime(file):
    with st.sidebar:
        label = st.text_input("Label", value=file.name if not isinstance(file, str) else "Example")
        title = st.text_input("Plot title", value="Lifetime")
        xlim = st.slider("X-axis limit", 0.0, 50.0, (0.0, 25.0))
        ylim = st.slider("Y-axis limit", 0.0, 2000.0, (0.0, 2000.0))

    if isinstance(file, str):  # Demo mode
        data = pd.read_csv(pd.compat.StringIO(file), delimiter='\t', header=0)
    else:
        data = pd.read_csv(file, delimiter='\t', header=0)

    fig = px.line(data, x="Time (s)", y="Lifetime", title=title, labels={'value': label})
    fig.update_layout(xaxis_range=xlim, yaxis_range=ylim)
    st.plotly_chart(fig)
