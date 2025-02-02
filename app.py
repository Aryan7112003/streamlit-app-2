import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 🔹 Step 1: Download Model from Google Drive (If Not Exists)
MODEL_FILE = "best_vgg16_model.keras"
GOOGLE_DRIVE_FILE_ID = "1wB0qSn_9CAsl7tdq0RDVi03VbQQfMqfj"  # ✅ Your Google Drive File ID

def download_model():
    if not os.path.exists(MODEL_FILE):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        st.info("Downloading model... Please wait.")
        response = requests.get(url)
        with open(MODEL_FILE, "wb") as file:
            file.write(response.content)

download_model()

# 🔹 Step 2: Load the Model
try:
    model = load_model(MODEL_FILE)
    st.success("Model loaded successfully! ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 🔹 Step 3: Class Information
class_info = {
    "test": {
        "description": "This is a test class, typically used for validation.",
        "country": "United States",
        "rank": 1,
        "advantages": "Test datasets help ensure that models are generalizable and accurate.",
        "disadvantages": "It may not cover all edge cases, leading to overfitting or underfitting."
    },
    "train": {
        "description": "This is a training class used for model training.",
        "country": "India",
        "rank": 2,
        "advantages": "Training data helps the model learn and improve its performance.",
        "disadvantages": "Training data can be biased, leading to inaccurate model predictions in real-world scenarios."
    }
}

# 🔹 Step 4: Streamlit UI
st.title("Image Classification with Advantages and Disadvantages")
uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).resize((224, 224)).convert("RGB")
        images.append(img_to_array(image) / 255.0)  # Preprocess and normalize images
    
    # Convert list of images to numpy array
    image_array = np.array(images)

    # Predict class for each image in the batch
    predictions = model.predict(image_array)
    predicted_classes = [np.argmax(pred) for pred in predictions]

    # Display results
    for i, uploaded_file in enumerate(uploaded_files):
        predicted_class_index = predicted_classes[i]

        # Check if the predicted class index is valid
        if predicted_class_index < len(class_info):
            predicted_class_name = list(class_info.keys())[predicted_class_index]  # Get class name
            class_details = class_info[predicted_class_name]

            st.image(uploaded_file, caption=f"Uploaded Image {i+1}")
            st.write(f"Prediction: {predicted_class_name}")
            st.write(f"Description: {class_details['description']}")
            st.write(f"Country Mostly Found: {class_details['country']}")
            st.write(f"Rank: {class_details['rank']}")
            st.write(f"**Advantages:** {class_details['advantages']}")
            st.write(f"**Disadvantages:** {class_details['disadvantages']}")
        else:
            st.image(uploaded_file, caption=f"Uploaded Image {i+1}")
            st.write("Prediction: Unknown")
            st.write("The predicted class index is out of range.")
