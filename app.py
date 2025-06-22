import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

# Constants
MODEL_PATH = "best_model.h5"
LABEL_PATH = "class_names.txt"
IMAGE_SIZE = 64

# Title and description
st.title("🌿 Plant Disease Detection")
st.markdown("Upload a plant leaf image and check if it has a disease.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess image
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize to 64x64
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    try:
        # Load model
        model = load_model(MODEL_PATH)

        # Load class names
        with open(LABEL_PATH, "r") as f:
            class_names = [line.strip() for line in f.readlines()]

        # Predict
        predictions = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Show result
        st.markdown(f"🧠 **Prediction:** `{predicted_class}`")
        st.markdown(f"🔍 **Confidence:** `{confidence:.2f}%`")

        if "Healthy" not in predicted_class:
            st.warning("⚠️ The plant shows signs of disease.")
        else:
            st.success("✅ The plant looks healthy.")

    except Exception as e:
        st.error("❌ Failed to load model or predict.")
        st.text(str(e))
