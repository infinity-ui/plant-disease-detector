import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

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
    image = image.resize((224, 224))  # Resize for CNN input
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Load model
    try:
        model = load_model("cnn_model.h5")
        predictions = model.predict(img_array)[0]
        class_names = ["Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Healthy"]

        # Get top prediction
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Show prediction
        st.markdown(f"🧠 **Prediction:** `{predicted_class}`")
        st.markdown(f"🔍 **Confidence:** `{confidence:.2f}%`")

        if "Healthy" not in predicted_class:
            st.warning("⚠️ The plant shows signs of disease.")
        else:
            st.success("✅ The plant looks healthy.")

    except Exception as e:
        st.error("❌ Failed to load model or predict.")
        st.text(str(e))
