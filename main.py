from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import pandas as pd

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:/Users/rudza/PycharmProjects/ML/keras_model.h5", compile=False)

# Load the labels
class_names = open("C:/Users/rudza/PycharmProjects/ML/labels.txt", "r").readlines()

def predict_image(image):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the class of the image
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def main():
    st.title("Image Classification App")
    uploaded_file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        class_name, confidence_score = predict_image(image)
        st.write("Class:", class_name)
        st.write("Confidence Score:", confidence_score)

if __name__ == '__main__':
    main()
