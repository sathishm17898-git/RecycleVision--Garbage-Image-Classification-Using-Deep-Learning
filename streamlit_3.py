import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model =  tf.keras.models.load_model("D:/Sathish/AIML/Garbage Classification/ResNet50_cnn.keras")
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

st.title("Garbage Image Classification")

pages = st.sidebar.radio(
    "Go to",
    ['Home','Image Classification']
)
if pages == "Home":
    st.write("RecycleVision- Garbage Image Classification Using Deep Learning")
    st.write(""" It classifies the garbage images as "cardboard", "glass", "metal", "paper", "plastic", "trash".
             The model built using the Convolution Neural Network: ResNet50 model.
             Upload the images and predict the class
""")

else:
    upload_file = st.file_uploader("Upload an image",type=['jpg','png','jpeg'])
    if upload_file is not None:
        image = Image.open(upload_file).convert('RGB')
        st.image(image,caption='uploaded image')
        img_resize = image.resize((224,224))
        img_array = np.array(img_resize)
        img_array = np.expand_dims(img_array,axis=0)
        img_array = tf.keras.applications.resnet.preprocess_input(img_array)

        #Prediction
        pred = model.predict(img_array)
        pred_class = class_names[np.argmax(pred)]
        confidence = np.max(pred)
        conf = confidence*100

        if st.button("Predict"):
            #Display results
            st.write(f"Predicted class:{pred_class}")
            st.write(f"Predicted confidence level:{conf:.2f}%")

        