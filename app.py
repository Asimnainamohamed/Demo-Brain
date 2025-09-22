# streamlit_app.py
import streamlit as st
import os
import random
from PIL import Image

# Title
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠")
st.title("🧠 Brain Tumor Classifier (Demo)")

# Allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Class labels (for demonstration)
class_labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Upload section
uploaded_file = st.file_uploader("Upload an MRI image", type=list(ALLOWED_EXTENSIONS))

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Prediction button
    if st.button("Predict"):
        try:
            # Simulate prediction (replace with ML model later)
            predicted_class = random.choice(class_labels)
            confidence = random.uniform(85.0, 99.9)

            # Show result
            st.success(f"🎯 Prediction: **{predicted_class.replace('_', ' ').title()}**")
            st.info(f"📊 Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
else:
    st.warning("Please upload a PNG/JPG/JPEG image to continue.")

