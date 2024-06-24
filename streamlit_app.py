import streamlit as st
from count_rate import plot_count_rate
from g2 import plot_g2
from lifetime import plot_lifetime
import os

def load_example_data(plot_type):
    example_data_path = os.path.join('example data', f'example {plot_type}.txt')
    with open(example_data_path, 'r') as file:
        return file.read()

st.title("Data Plotting Application")

plot_type = st.selectbox("Select what to plot/fit", ["Count Rate", "g2", "Lifetime"])
demo_mode = st.checkbox("Demo mode")

if demo_mode:
    st.write("Demo mode selected. Loading example data...")
    if plot_type == "Count Rate":
        uploaded_file = load_example_data("count rate")
    elif plot_type == "g2":
        uploaded_file = load_example_data("g2")
    elif plot_type == "Lifetime":
        uploaded_file = load_example_data("lifetime")
else:
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")

if uploaded_file:
    if plot_type == "Count Rate":
        plot_count_rate(uploaded_file)
    elif plot_type == "g2":
        plot_g2(uploaded_file)
    elif plot_type == "Lifetime":
        plot_lifetime(uploaded_file)
