import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("best_model.h5")

# Load class names
@st.cache_data
def load_class_names():
    with open("class_names.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

model = load_trained_model()
class_names = load_class_names()
IMAGE_SIZE = 64

# UI
st.set_page_config(page_title="🌿 Plant Disease Detector")
st.title("🌿 Plant Disease Detection")
st.markdown("Upload a plant leaf image and check if it has a disease.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    predicted_label = class_names[np.argmax(prediction)]

    st.markdown(f"### 🧠 Prediction: `{predicted_label}`")
    st.markdown(f"### 🔍 Confidence: `{confidence * 100:.2f}%`")

    if "healthy" in predicted_label.lower():
        st.success("🍀 The plant appears **healthy**.")
    else:
        st.warning("⚠️ The plant shows signs of **disease**.")
else:
    st.info("📂 Please upload a leaf image to get started.")
