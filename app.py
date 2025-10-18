from flask import Flask, render_template, request, jsonify
import io
import torch
from src.model import get_vgg16_model
from src.predict import preprocess_image, predict_from_model

app = Flask(__name__)

# Configuration - adjust paths as needed
CHECKPOINT = "vgg16_best_valacc.pth"
NUM_CLASSES = None  # will be determined when loading model if possible


def load_model(checkpoint=CHECKPOINT, num_classes=4, device=torch.device('cpu')):
    model = get_vgg16_model(num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    except Exception:
        # If checkpoint missing, return freshly initialized model
        pass
    return model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'no image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # For demo, assume 4 classes; in production store classes mapping
    model = load_model(device=device, num_classes=NUM_CLASSES or 4)
    img_t = preprocess_image(img_bytes)
    results = predict_from_model(model, img_t, device=device, classes=None)
    return jsonify({'predictions': results})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
