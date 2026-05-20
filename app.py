import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model(os.path.join("models", "plant_disease_model.h5"))

# Class labels
dataset_path = os.path.join("Dataset", "PlantVillage")
if os.path.isdir(dataset_path):
    classes = sorted([
        d.replace("_", " ").title()
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])
else:
    classes = [
        "Pepper Bell Healthy",
        "Potato Early Blight",
        "Potato Late Blight",
        "Potato Healthy",
        "Tomato Early Blight",
        "Tomato Late Blight",
        "Tomato Healthy"
    ]

# Page title
st.title("🌿 Plant Disease Detection System")

st.write("Upload a plant leaf image to detect disease.")

# File upload
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize image
    img = image.resize((128,128))

    # Convert to numpy array
    img_array = np.array(img)

    # Normalize image
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    # Show results
    st.success(f"Prediction: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")

    # Disease suggestions
    if "Blight" in predicted_class:
        st.warning("⚠️ Disease detected. Remove infected leaves and use proper fungicide treatment.")

    else:
        st.success("✅ Plant appears healthy.")