import streamlit as st
import cv2
import time

# -------------------------------------------------
# Application
# -------------------------------------------------
st.markdown("# Try DeepASL now 🧏")
st.sidebar.markdown("# About the DeepASL App 🧏")
st.sidebar.caption("The DeepASL App is upgraded with a new dataset, a new model (trained using Convolutional Neural Networks), modified code, and a user interface.")
st.caption("This app would like to use your camera 📷 Click on the button when ready!")

video_capture = cv2.VideoCapture(0)


if st.button("Launch the App"):
    
    #Test if camera is opened

    if not video_capture.isOpened():
        st.warning("⚠️Alert: Camera access is required to use this app. Please enable it first!")

    else:
        st.balloons()
        st.write("Camera OK!")

        with st.spinner("Loading app..."):
            time.sleep(5)
        st.success("Done!")





