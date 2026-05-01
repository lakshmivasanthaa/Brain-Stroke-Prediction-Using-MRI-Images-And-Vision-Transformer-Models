import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from timm import create_model
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load classes
classes = ["Normal", "Haemorrhagic"]

# Data Preprocessing
image_size = (224, 224)

# Define preprocessing with padding to maintain aspect ratio
def preprocess_image():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform

# Load Vision Transformer model
def load_model(num_classes):
    model = create_model("vit_small_patch16_224", pretrained=False, num_classes=num_classes)
    weights_path = os.path.join(os.path.dirname(__file__), "vit_model_weights.pth")
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))) #Load weights
    model.eval()
    return model

# Function to predict a new image
def predict_image(image, model):
    device = torch.device("cpu")  # Use CPU for streamlit
    model.to(device)
    model.eval()

    transform = preprocess_image()
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted_class = torch.max(output, 1)

    predicted_label = classes[predicted_class.item()]
    if predicted_label== "Normal":
        predicted_label = "Haemorrhagic"
    else:
        predicted_label = "Normal"
    return predicted_label

def main():
    st.title("Stroke Detection using Vision Transformer")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        if st.button("Predict"):
            try:
                model = load_model(len(classes))
                prediction = predict_image(image, model)
                st.write(f"Predicted Class: {prediction}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()