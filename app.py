import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.cnn import resnet50
from utils.dataset import create_data_loader, get_transforms
from utils.train_eval import evaluate

model = resnet50()
model.load_state_dict(torch.load('best_model.pth'))
_, class_names = create_data_loader(".\\data\\val", batch_size=1, is_train=False)

uploaded_file = st.file_uploader("Wgraj obrazek", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Wgrany obrazek", use_column_width=True)

    transform = get_transforms(train=False)
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_index = torch.argmax(probabilities).item()
    st.write(f"Predykcja: {class_names[predicted_index]} (prawdopodobieństwo: {probabilities[predicted_index]:.2f})")
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    st.write(f"Top 5 predykcji: ")
    for i in range(5):
        st.write(f"{class_names[top5_idx[i]]}: {top5_prob[i]:.4f}")