import streamlit as st
import tensorflow as tf
import os

@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('model_custom.h5')
    return model

def get_recommendation(result_class):
    if result_class == 0:
        return "Please wear your mask properly."
    elif result_class == 1:
        return "Your mask is detected and worn correctly. Keep wearing it."
    elif result_class == 2:
        return "Please wear a mask for your safety."

model = load_model()

st.write("# Mask Detection System")

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_path, model):
    size = (35, 35)
    image = Image.open(image_path)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_rgb = image.convert("RGB")
    img = np.asarray(image_rgb)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

image_dir = "./test_face/"
image_files = os.listdir(image_dir)

selected_image = st.selectbox("Select an image", image_files)

if selected_image:
    image_path = os.path.join(image_dir, selected_image)
    st.image(image_path, use_column_width=True)
    prediction = import_and_predict(image_path, model)
    class_names = {0: 'MASK INCORRECT', 1: 'MASK', 2: 'NO MASK'}
    result_class = np.argmax(prediction)
    string = "OUTPUT : " + class_names[result_class]
    st.success(string)
    st.write("## Recommendation")
    st.info(get_recommendation(result_class))
