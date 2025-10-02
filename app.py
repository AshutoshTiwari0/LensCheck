import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import requests


# Function to build model architecture
def build_model():
    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu', input_shape=(224,224,3)))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model


# Download weights if not present
weights_url = "https://github.com/AshutoshTiwari0/LensCheck/releases/download/v1.0/cataract_model.weights.h5"
weights_path = "cataract_model.weights.h5"

if not os.path.exists(weights_path):
    st.text("Downloading model weights... please wait")
    r = requests.get(weights_url)
    with open(weights_path, "wb") as f:
        f.write(r.content)
    st.text("Download complete!")

# Build and load model
model = build_model()
model.load_weights(weights_path)


# Streamlit UI
st.title('LensCheck - A CNN based cataract detection model')

file = st.file_uploader("Please upload an image", type=['jpg','png','jpeg'])

st.text('Go to GitHub for sample images')


# Prediction function
def import_and_predict(image_data, model):
    size = (224,224)
    image = ImageOps.fit(image_data, size)
    image = np.asarray(image) / 255.0
    img = np.expand_dims(image, axis=0)
    prediction = model.predict(img)
    return prediction


# Handle uploaded file
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_container_width=True)   
    predictions = import_and_predict(image, model)

    # threshold at 0.5
    if predictions[0][0] > 0.5:
        st.header("Normal Detected")
    else:
        st.header("Cataract Detected")
