"""
Simple Flask app for Alzheimer's MRI classification using VGG16.

Setup:
1. Ensure models/ folder exists at repo root with:
   - vgg16_best_valacc.pth
   - class_names.json
2. Create virtual environment: python -m venv venv
3. Activate: venv\Scripts\activate (Windows) or source venv/bin/activate (Linux/Mac)
4. Install dependencies: pip install -r requirements.txt
5. Run: python app.py
6. Open browser: http://localhost:5000
"""

import json
import io
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
from flask import Flask, request, jsonify, render_template, send_from_directory

# Configuration
MODEL_PATH = 'models/vgg16_best_valacc.pth'
CLASS_NAMES_PATH = 'resources/class_names.json'
MAX_UPLOAD_SIZE = 8 * 1024 * 1024  # 8MB

# Global variables
model = None
class_names = None
device = None


def load_class_names(path):
    """Load class names from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def create_vgg16_model(num_classes):
    """Recreate VGG16 architecture exactly as trained."""
    # Use weights arg to avoid deprecation warnings and match training
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Freeze early layers (first 20 conv blocks), keep later convs trainable
    for param in model.features[:20].parameters():
        param.requires_grad = False
    for param in model.features[20:].parameters():
        param.requires_grad = True

    # Replace classifier to match training notebook
    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


def load_model():
    """Initialize model and load trained weights."""
    global model, class_names, device
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load class names
    class_names = load_class_names(CLASS_NAMES_PATH)
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {class_names}")
    
    # Create model architecture
    model = create_vgg16_model(num_classes)
    
    # Load trained weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    
    # Set to evaluation mode
    model.to(device)
    model.eval()
    print("Model loaded successfully!")


def preprocess_image(image_bytes):
    """
    Preprocess image to match training preprocessing.
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Preprocessed tensor of shape (1, 3, 224, 224)
    """
    # Open image from bytes
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert grayscale to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to 224x224 using BILINEAR
    img = img.resize((224, 224), Image.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Transpose from HWC to CHW
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(img_array).unsqueeze(0)
    
    return tensor


def predict_image(image_bytes):
    """
    Run inference on image.
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    img_tensor = preprocess_image(image_bytes).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
    
    # Get probabilities as numpy array
    probs = probabilities.cpu().numpy()[0]
    
    # Create probabilities dictionary
    probs_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    
    # Get top 3 predictions
    top3_indices = np.argsort(probs)[-3:][::-1]
    top3 = [
        {
            'class': class_names[idx],
            'probability': float(probs[idx])
        }
        for idx in top3_indices
    ]
    
    return {
        'predicted_index': predicted_idx,
        'predicted_class': class_names[predicted_idx],
        'confidence': float(probs[predicted_idx]),
        'probabilities': probs_dict,
        'top3': top3
    }


# Create Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE


@app.route('/')
def index():
    """Landing page with dataset overview and project summary."""
    return render_template('index.html')


@app.route('/model')
def model_page():
    """Model details page."""
    return render_template('model.html')


@app.route('/predict')
def predict_page():
    """Prediction UI page."""
    return render_template('predict.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Handle image upload and return prediction.
    
    Expected: multipart/form-data with 'image' file
    Returns: JSON with prediction results
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Validate image
        if len(image_bytes) == 0:
            return jsonify({'error': 'Empty file'}), 400
        
        # Run prediction
        result = predict_image(image_bytes)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 8MB'}), 413


@app.route('/download-notebook')
def download_notebook():
    """Allow users to download the training notebook used for the model."""
    notebook_filename = "Alzhemer's_VGG16_.ipynb"
    return send_from_directory(
        'resources',
        notebook_filename,
        as_attachment=True,
        download_name='alzheimers_vgg16_training.ipynb'
    )


@app.route('/download-model')
def download_model():
    """Allow users to download the trained model weights."""
    model_filename = 'vgg16_best_valacc.pth'
    return send_from_directory(
        'models',
        model_filename,
        as_attachment=True,
        download_name='alzheimers_vgg16_best.pth'
    )


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    print("\nStarting Flask app...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
