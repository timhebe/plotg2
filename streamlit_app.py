import streamlit as st
from count_rate import plot_count_rate
from g2 import plot_g2
from lifetime import plot_lifetime
import os

def load_example_data(plot_type):
    example_data_path = os.path.join('example data', f'example {plot_type}.txt')
    with open(example_data_path, 'rb') as file:
        return file

st.title("Data Plotting Application")

plot_type = st.selectbox("Select what to plot/fit", ["Count Rate", "g2", "Lifetime"])
demo_mode = st.checkbox("Demo mode")

if demo_mode:
    st.write("Demo mode selected. Loading example data...")
    if plot_type == "Count Rate":
        uploaded_files = [load_example_data("count rate")]
    elif plot_type == "g2":
        uploaded_files = [load_example_data("g2")]
    elif plot_type == "Lifetime":
        uploaded_files = [load_example_data("lifetime")]
else:
    uploaded_files = st.file_uploader("Choose .txt files", type="txt", accept_multiple_files=True)

if uploaded_files:
    if plot_type == "Count Rate":
        plot_count_rate(uploaded_files)
    elif plot_type == "g2":
        plot_g2(uploaded_files)
    elif plot_type == "Lifetime":
        plot_lifetime(uploaded_files)
