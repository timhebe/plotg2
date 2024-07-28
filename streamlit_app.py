import streamlit as st
from count_rate import plot_count_rate
from g2 import plot_g2
from lifetime import plot_lifetime
from spectrum import plot_spectrum
from camera import plot_cam

def main():
    st.title("Scientific Data Plotter")

    plot_type = st.selectbox("Select the type of plot", ["", "Count Rate", "g2", "Lifetime", "Spectrum", "CAM"])

    device_type = ""
    if plot_type in ["Count Rate", "g2", "Lifetime"]:
        device_type = st.selectbox("Select the device", ["Swabian Instruments", "PicoQuant"])
    elif plot_type == "Spectrum":
        device_type = st.selectbox("Select the device", ["Princeton Instruments", "Andor, Oxford Instruments"])

    if plot_type:
        demo_mode = st.checkbox("Demo mode")

    uploaded_file = None
    if demo_mode:
        if plot_type == "Count Rate":
            uploaded_file = "example_data/example_count_rate.txt"
        elif plot_type == "g2":
            uploaded_file = "example_data/example_g2.txt"
        elif plot_type == "Lifetime":
            uploaded_file = "example_data/example_lifetime.txt"
        elif plot_type == "Spectrum":
            if device_type == "Princeton Instruments":
                uploaded_file = "example_data/example_spectrum_princeton.txt"
            else:
                uploaded_file = "example_data/example_spectrum_andor.asc"
        elif plot_type == "CAM":
            uploaded_file = "example_data/example_camera.cam"
    else:
        uploaded_file = st.file_uploader("Upload your data file", type=["txt", "dat", "asc", "cam"])

    if plot_type and uploaded_file:
        if plot_type == "Count Rate":
            plot_count_rate(uploaded_file, device_type)
        elif plot_type == "g2":
            plot_g2(uploaded_file, device_type)
        elif plot_type == "Lifetime":
            plot_lifetime(uploaded_file, device_type)
        elif plot_type == "Spectrum":
            plot_spectrum(uploaded_file, device_type)
        elif plot_type == "CAM":
            plot_cam(uploaded_file)

if __name__ == "__main__":
    main()
