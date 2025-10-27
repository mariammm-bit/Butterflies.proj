import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------- Load Model ----------
MODEL_PATH = r"D:\Butterflies.proj\Models\butterfly_classifier_mobilenet.h5"
model = tf.keras.models.load_model(MODEL_PATH)


class_names = [
    "class1", "class2", "class3", "class4"  
]

# ---------- Streamlit UI ----------
st.title("Butterfly Classifier")
st.write("Upload an image of a butterfly, and the model will predict its class.")

uploaded_file = st.file_uploader("pick aphoto", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="the uploaded image is", use_column_width=True)

    
    img_size = (224, 224)
    img_array = tf.keras.utils.img_to_array(image.resize(img_size))
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)
    confidence = prediction[0][pred_class]

    
    st.subheader(f"Predication : {class_names[pred_class]}")
    st.write(f"Confidence_Score : {confidence*100:.2f}%")  