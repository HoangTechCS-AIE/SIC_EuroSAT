# api_eurosat.py
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import base64

app = FastAPI()

LABELS = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

# Định nghĩa lại mô hình y hệt lúc training
def build_model():
    model = models.vgg19(weights=None)
    model.avgpool = nn.AdaptiveAvgPool2d((1,1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )
    return model

model = build_model()
model.load_state_dict(torch.load("models/trained_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class ImageRequest(BaseModel):
    image_base64: str

@app.post("/predict")
def predict(req: ImageRequest):
    try:
        image_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(x)
            pred = output.argmax(dim=1).item()

        return {"index": pred, "label": LABELS[pred]}

    except Exception as e:
        return {"error": str(e)}
