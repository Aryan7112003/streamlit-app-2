import streamlit as st
import numpy as np
from PIL import Image

# 🔹 Step 1: Mock Classification - No TensorFlow
# Here, we can use a simple mock prediction for the sake of demonstrating image classification.
# Replace this with your own classification logic.
def mock_predict(image_array):
    # Mock prediction: Randomly choose between "test" or "train"
    return np.random.choice(["test", "train"])

# 🔹 Step 2: Class Information
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

# 🔹 Step 3: Streamlit UI
st.title("Image Classification with Advantages and Disadvantages")
uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).resize((224, 224)).convert("RGB")
        images.append(np.array(image))  # Convert image to numpy array
    
    # Mock predictions for each image
    predictions = [mock_predict(img) for img in images]

    # Display results
    for i, uploaded_file in enumerate(uploaded_files):
        predicted_class = predictions[i]

        # Check if the predicted class index is valid
        if predicted_class in class_info:
            class_details = class_info[predicted_class]

            st.image(uploaded_file, caption=f"Uploaded Image {i+1}")
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Description: {class_details['description']}")
            st.write(f"Country Mostly Found: {class_details['country']}")
            st.write(f"Rank: {class_details['rank']}")
            st.write(f"**Advantages:** {class_details['advantages']}")
            st.write(f"**Disadvantages:** {class_details['disadvantages']}")
        else:
            st.image(uploaded_file, caption=f"Uploaded Image {i+1}")
            st.write("Prediction: Unknown")
            st.write("The predicted class index is out of range.")
