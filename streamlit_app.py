import streamlit as st
from count_rate import plot_count_rate
from g2 import plot_g2
from lifetime import plot_lifetime
from spectrum import plot_spectrum
from camera import plot_cam


def add_footer():
    st.markdown(
        """
        <style>
        footer {
            visibility: hidden;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: #555555;
            text-align: center;
            padding: 10px;
            font-size: 12px;
        }
        </style>
        <div class="footer">
            <p>Created by Tim Hebenstreit. For any questions, please contact me at: 
            <a href="mailto:tim.hebenstreit@mpl.mpg.de">tim.hebenstreit@mpl.mpg.de</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    st.title("Scientific Data Plotter")

    plot_type = st.selectbox("Select the type of plot",
                             ["Choose one from the list below...", "Count Rate", "g²(τ)",
                              "Lifetime", "Spectrum", "CAM"])

    device_type = ""
    if plot_type in ["Count Rate", "g²(τ)", "Lifetime"]:
        device_type = st.selectbox("Select the device", ["Swabian Instruments", "PicoQuant"])
    elif plot_type == "Spectrum":
        device_type = st.selectbox("Select the device", ["Princeton Instruments", "Andor, Oxford Instruments"])

    if plot_type:
        demo_mode = st.checkbox("Demo mode")

    uploaded_file = None
    if demo_mode:
        if plot_type == "Count Rate":
            if device_type == "Swabian Instruments":
                uploaded_file = "example_data/example count rate Swabian.txt"
            else:
                st.write("No demo data available for PicoQuant. Choose Swabian Instruments device or another plot.")
        elif plot_type == "g²(τ)":
            if device_type == "Swabian Instruments":
                uploaded_file = "example_data/example g2 Swabian.txt"
            else:
                uploaded_file = "example_data/example g2 PicoQuant.dat"
        elif plot_type == "Lifetime":
            if device_type == "Swabian Instruments":
                uploaded_file = "example_data/example lifetime Swabian.txt"
            else:
                st.write("No demo data available for PicoQuant. Choose Swabian Instruments device or another plot.")
        elif plot_type == "Spectrum":
            if device_type == "Princeton Instruments":
                uploaded_file = "example_data/example spectrum PI.csv"
            else:
                uploaded_file = "example_data/example spectrum Andor.asc"
        elif plot_type == "CAM":
            uploaded_file = "example_data/example camera.cam"
    else:
        uploaded_file = st.file_uploader("Upload your data file", type=["txt", "dat", "asc", "cam", "csv"])

    if plot_type and uploaded_file:
        if plot_type == "Count Rate":
            plot_count_rate(uploaded_file, device_type)
        elif plot_type == "g²(τ)":
            plot_g2(uploaded_file, device_type)
        elif plot_type == "Lifetime":
            plot_lifetime(uploaded_file, device_type)
        elif plot_type == "Spectrum":
            plot_spectrum(uploaded_file, device_type)
        elif plot_type == "CAM":
            plot_cam(uploaded_file)

    add_footer()


if __name__ == "__main__":
    main()
