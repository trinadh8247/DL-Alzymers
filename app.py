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
MODEL_AUTH_TOKEN = os.getenv('MODEL_AUTH_TOKEN')
NUM_CLASSES = int(os.getenv('NUM_CLASSES', '4'))

logger = get_logger('app')

# Global model (load once on startup)
MODEL = None
MODEL_LOADED = False


def try_download_model(checkpoint=CHECKPOINT, model_url=MODEL_URL, auth_token=MODEL_AUTH_TOKEN):
    if os.path.exists(checkpoint):
        logger.info(f"Checkpoint already exists at {checkpoint}")
        return
    if not model_url:
        logger.warning("No MODEL_URL provided and checkpoint missing")
        return
    try:
        logger.info(f"Downloading model from {model_url} to {checkpoint}")
        headers = {}
        if auth_token:
            headers['Authorization'] = f"Bearer {auth_token}"
        resp = requests.get(model_url, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()
        with open(checkpoint, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info("Model downloaded successfully")
    except Exception as e:
        logger.exception("Failed to download model")
        raise AppException("Model download failed", e)


def load_model_into_memory(checkpoint=CHECKPOINT, num_classes=NUM_CLASSES, device=None):
    global MODEL, MODEL_LOADED
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = get_vgg16_model(num_classes=num_classes)
        if os.path.exists(checkpoint):
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            logger.info(f"Loaded checkpoint from {checkpoint}")
            MODEL = model.to(device)
            MODEL_LOADED = True
        else:
            logger.warning(f"Checkpoint {checkpoint} not found; model will not be loaded into memory")
            MODEL = model.to(device)
            MODEL_LOADED = False
    except Exception as e:
        logger.exception("Failed to load model checkpoint into memory")
        MODEL = None
        MODEL_LOADED = False
        raise AppException("Failed to load model", e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL_LOADED
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'no image uploaded'}), 400

        file = request.files['image']
        img_bytes = file.read()

        if MODEL is None:
            # Attempt to (re)load model into memory
            load_model_into_memory()
        if MODEL is None:
            return jsonify({'error': 'model not available'}), 503

        device = next(MODEL.parameters()).device if MODEL is not None else torch.device('cpu')
        img_t = preprocess_image(img_bytes)
        results = predict_from_model(MODEL, img_t, device=device, classes=None)
        return jsonify({'predictions': results})
    except AppException as e:
        logger.exception("Prediction endpoint failed")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.exception("Unexpected error in prediction endpoint")
        return jsonify({'error': 'internal server error'}), 500


def initialize_service():
    try:
        try_download_model()
    except AppException:
        logger.warning("Model download failed during init; continuing and will attempt to load if available")
    try:
        load_model_into_memory()
    except AppException:
        logger.warning("Model failed to load into memory at startup")


if __name__ == '__main__':
    initialize_service()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=os.getenv('FLASK_DEBUG', 'False') == 'True')
