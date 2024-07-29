import streamlit as st
import matplotlib.pyplot as plt
from CamReader import CamReader
import io

def plot_cam(file):
    if isinstance(file, str):  # Demo mode
        name = file.split('/')[-1].split('.')[0]
        file_path = file
    else:
        name = file.name.split('.')[0]
        file_bytes = file.read()
        file_path = None  # No file path available for uploaded file

    try:
        if file_path:
            CAM = CamReader(file_path, same_size=True)
        else:
            # Save the in-memory file to a temporary file
            temp_file_path = f"/tmp/{file.name}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_bytes)
            CAM = CamReader(temp_file_path, same_size=True)

        plt.imshow(CAM[0])
        plt.title("Cryo Measurement")
        st.pyplot()

        buf = io.BytesIO()
        plt.savefig(buf, format="pdf")
        st.download_button("Download as PDF", buf.getvalue(), file_name=f"{name}.pdf")
        buf.seek(0)
        plt.savefig(buf, format="png")
        st.download_button("Download as PNG", buf.getvalue(), file_name=f"{name}.png")

    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
