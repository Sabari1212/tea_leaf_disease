import streamlit as st
import cv2
import numpy as np

from PIL import Image
import pickle
from fastapi import FastAPI,UploadFile,File
from fastapi.responses import JSONResponse,HTMLResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

# Load your pre-trained model
with open('tea_model','rb') as f:
    model=pickle.load(f)

def image_to_array(data):
    return np.array(Image.open(BytesIO(data)))

# Function to preprocess the image and make predictions
def preprocess_and_predict(image):
    # # Convert the image to an array
    img_array = np.array(image)
    image_batch=np.expand_dims(img_array,0)
    prediction=model.predict(image_batch)
   
    # # Resize the image to match the input shape the model expects
    # img_resized = cv2.resize(img_array, (256, 256))  # assuming your model expects 224x224 images
    # img_resized = np.expand_dims(img_resized, axis=0)  # add batch dimension
    # # Normalize the image
    # img_normalized = img_resized / 255.0
    # # Make predictions
    # prediction = model.predict(img_normalized)
    
        
    return prediction

# Streamlit app
st.title('Tea Leaf Disease Identification')
st.write("Upload an image of a tea leaf to identify the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess and predict
    st.write("Classifying...")
    prediction = preprocess_and_predict(image)
    
    # Display the result
    st.write(f"Prediction: {prediction}")
    
    # Here you can add logic to interpret the prediction and display the specific disease
    # For example, if using a softmax output, you can display the class with the highest probability
    # Assuming 'classes' is a list of disease names corresponding to the model's output classes
    classes = ["algal_spot","brown_blight","gray_blight","healthy","helopeltis","red_spot"]  # Replace with actual class names
    predicted_class=classes[np.argmax(prediction[0])]
    confidence=np.max(prediction[0])
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence}")
    
