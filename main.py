import streamlit as st
import tensorflow as tf
import os

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model_custom.h5')
  return model
model=load_model()
st.write("""
# Mask Detection System"""
)
file=st.file_uploader("Choose plant photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_path,model):
    size=(35,35)
    image = Image.open(image_path)
    image=ImageOps.fit(image,size,Image.ANTIALIAS)
    image_rgb = image.convert("RGB")
    img = np.asarray(image_rgb)
    img_reshape=img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction
  
image_dir = "./test_face/"
image_files = os.listdir(image_dir)  
  
selected_image = st.selectbox("Select an image", image_files)

if selected_image:
    image_path = os.path.join(image_dir, selected_image)
    st.image(image_path,use_column_width=True)
    prediction=import_and_predict(image_path,model)
    class_names={0:'MASK INCORRECT',1:'MASK', 2:'NO MASK'}
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
