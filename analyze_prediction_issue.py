"""
Analyze prediction issues by comparing training vs inference preprocessing
and checking for class bias
"""
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io

# Configuration
MODEL_PATH = 'models/vgg16_best_valacc.pth'
CLASS_NAMES_PATH = 'models/class_names.json'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load class names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

print("="*70)
print("ANALYZING PREDICTION ISSUE")
print("="*70)
print(f"\nClass names: {class_names}")
print(f"Total classes: {len(class_names)}")

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

# Check what preprocessing was used in training vs app
print("\n" + "="*70)
print("PREPROCESSING COMPARISON")
print("="*70)

print("\nTraining preprocessing (from notebook):")
print("  - transforms.Resize((224, 224))")
print("  - transforms.ToTensor()")
print("  - transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])")

print("\nFlask app preprocessing (from app.py):")
print("  - img.resize((224, 224), Image.BILINEAR)")
print("  - Manual normalize: (img_array - mean) / std")
print("  - Note: Image.BILINEAR is equivalent to PIL's Resampling.BILINEAR")

# Test with synthetic images to check for output bias
print("\n" + "="*70)
print("TESTING MODEL BIAS (synthetic inputs)")
print("="*70)

# Create synthetic inputs
test_cases = [
    ("All zeros (black)", torch.zeros(1, 3, 224, 224)),
    ("All ones (white)", torch.ones(1, 3, 224, 224)),
    ("Random normal", torch.randn(1, 3, 224, 224)),
    ("ImageNet mean", torch.full((1, 3, 224, 224), 0.5)),
]

with torch.no_grad():
    for name, tensor in test_cases:
        tensor = tensor.to(device)
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_idx].item()
        
        print(f"\n{name}:")
        print(f"  Predicted: {class_names[pred_idx]} ({pred_prob:.4f})")
        
        # Show distribution
        sorted_indices = torch.argsort(probs[0], descending=True)
        print("  All probabilities:")
        for idx in sorted_indices:
            print(f"    {class_names[idx]:25s}: {probs[0, idx].item():.4f}")

# Check if model has learned anything by looking at gradients
print("\n" + "="*70)
print("MODEL WEIGHT ANALYSIS")
print("="*70)

# Get final classifier weights
final_layer = model.classifier[6][3]  # The Linear(512, 4) layer
weights = final_layer.weight.data
bias = final_layer.bias.data

print(f"\nFinal classification layer (Linear(512, 4)):")
print(f"  Weight shape: {weights.shape}")
print(f"  Bias shape: {bias.shape}")
print(f"\nBias values (determines class prior):")
for i, (name, b) in enumerate(zip(class_names, bias)):
    print(f"  {i}: {name:25s} -> {b.item():.4f}")

print(f"\nWeight statistics:")
for i, name in enumerate(class_names):
    w = weights[i]
    print(f"  {i}: {name:25s} -> mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}")

# Most important: check if model is just using bias to predict
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

# Rank classes by bias
bias_sorted = sorted(zip(class_names, bias.tolist()), key=lambda x: x[1], reverse=True)
print("\nClasses ranked by bias (higher = more likely to be predicted):")
for i, (name, b) in enumerate(bias_sorted, 1):
    print(f"  {i}. {name:25s} (bias={b:.4f})")

# Check if the model is just defaulting to one class
max_bias = bias.max().item()
min_bias = bias.min().item()
bias_range = max_bias - min_bias

print(f"\nBias range: {bias_range:.4f}")
if bias_range < 0.5:
    print("  WARNING: Very small bias range - model may be near-uniform in its priors")
elif bias_range < 2.0:
    print("  CAUTION: Moderate bias range - some class imbalance may exist")
else:
    print("  OK: Good bias range suggesting class balance in training")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

# Based on user's feedback
print("""
Based on your observation that "Non Demented predicts well but others don't":

Likely causes:
1. CLASS IMBALANCE: Training data has unequal distribution
   - Some classes (likely "Very Mild Dementia") have many more samples
   - Model learned to bias toward that class
   
2. LABEL MISMATCH: Possible mismatch between folder names and class_names.json
   - Even though we fixed the order, folder names might differ from JSON names
   
3. TRAINING ISSUE: Model may not have converged properly
   - The training notebook artificially inflates validation accuracy
   - This suggests real performance might be worse

SOLUTIONS TO TRY:
1. Check class distribution in training data (count images per folder)
2. Use class weights during training: weight = total_samples / (num_classes * class_samples)
3. Retrain with balanced sampler or data augmentation
4. Verify folder names match class names exactly (case-sensitive on Linux)
""")
