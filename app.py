import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import requests

# ----------------------------
# 🔧 Config
# ----------------------------
MODEL_PATH = "best_model.h5"
LABEL_PATH = "class_names.txt"
IMAGE_SIZE = 64
GROQ_API_KEY = "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"  # Replace this with your actual API key
GROQ_MODEL = "llama3-70b-8192"

# ----------------------------
# 🖼️ Title and Description
# ----------------------------
st.title("🌿 Plant Disease Detection")
st.markdown("Upload a plant leaf image and check if it has a disease.")

# ----------------------------
# 📤 File uploader
# ----------------------------
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        model = load_model(MODEL_PATH)

        with open(LABEL_PATH, "r") as f:
            class_names = [line.strip() for line in f.readlines()]

        predictions = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.markdown(f"🧠 **Prediction:** `{predicted_class}`")
        st.markdown(f"🔍 **Confidence:** `{confidence:.2f}%`")

        if "Healthy" not in predicted_class:
            st.warning("⚠️ The plant shows signs of disease.")
        else:
            st.success("✅ The plant looks healthy.")

    except Exception as e:
        st.error("❌ Failed to load model or predict.")
        st.text(str(e))

# ----------------------------
# 💬 Chatbot Section (Single-Turn)
# ----------------------------
st.header("💡 Ask About Plant Diseases")
st.markdown("Get remedies, treatment options, and care tips.")

# User input
user_query = st.text_input("Ask a question (e.g., 'How to treat late blight?')")

# Function to query Groq API
def query_groq(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Error: {response.text}"

# Only handle one message at a time (no history)
if user_query:
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in plant diseases and remedies."},
        {"role": "user", "content": user_query}
    ]
    reply = query_groq(messages)
    st.markdown(f"**👤 You:** {user_query}")
    st.markdown(f"**🤖 Bot:** {reply}")
