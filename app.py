from flask import Flask, render_template, request, jsonify
import io
import os
import requests
import torch
from src.model import get_vgg16_model
from src.predict import preprocess_image, predict_from_model
from src.logger import get_logger
from src.exception import AppException

app = Flask(__name__)

# Configuration - adjust paths as needed
CHECKPOINT = os.getenv('MODEL_PATH', 'vgg16_best_valacc.pth')
MODEL_URL = os.getenv('MODEL_URL')
NUM_CLASSES = int(os.getenv('NUM_CLASSES', '4'))

logger = get_logger('app')


def try_download_model(checkpoint=CHECKPOINT, model_url=MODEL_URL):
    if os.path.exists(checkpoint):
        logger.info(f"Checkpoint already exists at {checkpoint}")
        return
    if not model_url:
        logger.warning("No MODEL_URL provided and checkpoint missing")
        return
    try:
        logger.info(f"Downloading model from {model_url} to {checkpoint}")
        resp = requests.get(model_url, stream=True)
        resp.raise_for_status()
        with open(checkpoint, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Model downloaded successfully")
    except Exception as e:
        logger.exception("Failed to download model")
        raise AppException("Model download failed", e)


def load_model(checkpoint=CHECKPOINT, num_classes=4, device=torch.device('cpu')):
    model = get_vgg16_model(num_classes=num_classes)
    try:
        if os.path.exists(checkpoint):
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            logger.info(f"Loaded checkpoint from {checkpoint}")
        else:
            logger.warning(f"Checkpoint {checkpoint} not found; using untrained model")
    except Exception as e:
        logger.exception("Failed to load model checkpoint")
        raise AppException("Failed to load model", e)
    return model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'no image uploaded'}), 400

        file = request.files['image']
        img_bytes = file.read()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # For demo, assume NUM_CLASSES; in production store classes mapping
        model = load_model(device=device, num_classes=NUM_CLASSES)
        img_t = preprocess_image(img_bytes)
        results = predict_from_model(model, img_t, device=device, classes=None)
        return jsonify({'predictions': results})
    except AppException as e:
        logger.exception("Prediction endpoint failed")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.exception("Unexpected error in prediction endpoint")
        return jsonify({'error': 'internal server error'}), 500


if __name__ == '__main__':
    try:
        try_download_model()
    except AppException:
        logger.warning("Continuing without model download")
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=os.getenv('FLASK_DEBUG', 'False') == 'True')
