import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from code.model.convmixer import ConvMixer


# Load model
model_path = "models/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvMixer(dim=128, depth=4, kernel_size=8, patch_size=1, n_classes=10).to(device)

# Load the state_dict and filter out unwanted keys
state_dict = torch.load(model_path, map_location=device, weights_only=True)
filtered_state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
model.load_state_dict(filtered_state_dict)

# Define preprocessing function
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

st.title('ConvMixer Image Classification')

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run the model
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    st.write(f"Predicted Class: {predicted_class}")
