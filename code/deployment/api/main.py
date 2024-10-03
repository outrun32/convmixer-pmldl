import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from code.model.convmixer import ConvMixer

# Initialize FastAPI app
app = FastAPI()

# Load model from the models/ folder
model_path = "models/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your ConvMixer model architecture (adjust params if needed)
model = ConvMixer(dim=128, depth=4, kernel_size=8, patch_size=1, n_classes=10).to(device)
state_dict = torch.load(model_path, map_location=device, weights_only=True)
filtered_state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
model.load_state_dict(filtered_state_dict)

# Define preprocessing function
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Read the uploaded image file
    image = Image.open(file.file).convert('RGB')
    
    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run the model
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    return JSONResponse(content={"predicted_class": predicted_class})
