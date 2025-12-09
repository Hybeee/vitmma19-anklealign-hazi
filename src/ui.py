import streamlit as st
import numpy as np
import cv2

import requests
import io

API_URL = "http://127.0.0.1:8000/predict"

def main():
    st.set_page_config(page_title="AnkleAlign Client")

    st.title("AnkleAlign Client")
    st.caption(f"Connected to Inference Server: {API_URL}")
    st.write("Upload an image to classify it!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption='Uploaded image', width=300)

        if st.button("Classify Image"):
            with st.spinner("Sending to server..."):
                try:
                    uploaded_file.seek(0)

                    files = {"file": uploaded_file}

                    response = requests.post(API_URL, files=files)

                    if response.status_code == 200:
                        result = response.json()

                        if "error" in result:
                            st.error(f"Server Error: {result['error']}")
                        else:
                            prediction = result["prediction"]
                            confidence = result["confidence"]
                            probs = result["class_probabilities"]

                            st.divider()
                            st.header(f"Result: {prediction}")
                            st.metric(label="Confidence", value=confidence)

                            st.write("Class Probabilities:")
                            st.json(probs)
                    else:
                        st.error(f"HTTP Error: {response.status_code}: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to {API_URL}.")
                except Exception as e:
                    st.error(f"An error occured: {e}")

if __name__ == "__main__":
    main()