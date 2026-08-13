import json
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_DIR = Path(__file__).parent.parent/"models"

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_class_names():
    with open(MODEL_DIR/"class_names.json") as f:
        return json.load(f)
    
def load_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(MODEL_DIR/"best_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict(model, class_names, image: Image.Image, top_k: int = 3):
    image = image.convert("RGB")
    tensor = _transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
    top_probs, top_indices = probabilities.topk(top_k)
    
    return [
        {"class": class_names[i], "confidence": round(p.item(), 4)}
        for p, i in zip(top_probs, top_indices)
    ]