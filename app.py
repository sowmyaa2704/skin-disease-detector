import streamlit as st
import gdown
import os

MODEL_PATH = "skin_disease_mobilenet.h5"

# Download the model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download("https://drive.google.com/uc?id=1dVrmK3GdgguER3viJ4JDp2948M_AVwrW", MODEL_PATH, quiet=False)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model(MODEL_PATH)

# Your class labels (edit if needed)
class_labels = ['Acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Melanoma', 'Warts', 'Vitiligo', 'Cellulitis', 'Healthy']

st.title("Skin Disease Detection App")
st.write("Upload a skin image to predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((240, 240))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.markdown(f"### ?? Prediction: **{predicted_class}**")
    st.markdown(f"### ?? Confidence: **{confidence:.2f}**")

