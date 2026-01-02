"""
Diagnostic script to analyze per-class prediction accuracy
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

# Configuration
MODEL_PATH = 'models/vgg16_best_valacc.pth'
CLASS_NAMES_PATH = 'models/class_names.json'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load class names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)
print(f"\nClass names from JSON: {class_names}")

# Create model
def create_vgg16_model(num_classes):
    model = models.vgg16(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

model = create_vgg16_model(len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("\nModel loaded successfully!")

# Check the architecture
print("\nModel architecture:")
print(model)

# Test with a simple image to verify preprocessing
print("\n" + "="*60)
print("TESTING IMAGE PREPROCESSING")
print("="*60)

# Create test transforms (matching training validation transforms)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Check if there's a test image available
test_image_path = None
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_image_path = os.path.join(root, file)
            break
    if test_image_path:
        break

if test_image_path:
    print(f"\nFound test image: {test_image_path}")
    img = Image.open(test_image_path)
    print(f"Original image shape: {img.size}, mode: {img.mode}")
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Apply transforms
    img_tensor = test_transforms(img).unsqueeze(0).to(device)
    print(f"Transformed tensor shape: {img_tensor.shape}")
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
    
    print(f"\nPrediction results:")
    print(f"  Predicted index: {pred_idx}")
    print(f"  Predicted class: {class_names[pred_idx]}")
    print(f"\nAll class probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, probs[0].cpu().numpy())):
        print(f"  {i}: {name:25s} -> {prob:.4f}")
else:
    print("\nNo test image found in workspace")

print("\n" + "="*60)
print("CHECKING CLASS ORDER")
print("="*60)

# The key issue: what order does ImageFolder use?
print("\nAlphabetical order of classes (how ImageFolder would order them):")
sorted_classes = sorted(class_names)
for i, cls in enumerate(sorted_classes):
    print(f"  Index {i}: {cls}")

print("\nCurrent class_names.json order:")
for i, cls in enumerate(class_names):
    print(f"  Index {i}: {cls}")

print("\n⚠️  CRITICAL: These MUST match for correct predictions!")
print(f"Match: {sorted_classes == class_names}")
