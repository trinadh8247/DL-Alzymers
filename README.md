# Alzheimer's Disease Classification using VGG16

A deep learning web application for classifying Alzheimer's disease stages from MRI brain scans using a fine-tuned VGG16 convolutional neural network.

## Overview

This project implements an end-to-end solution for Alzheimer's disease classification, featuring:
- A trained VGG16 model fine-tuned for medical image classification
- A Flask web application with an intuitive user interface
- Real-time prediction with confidence scores
- Support for multiple stages of dementia classification

## Classification Categories

The model can classify MRI brain scans into four categories:
1. **Non Demented** - No signs of cognitive decline
2. **Very Mild Dementia** - Early stage cognitive impairment
3. **Mild Dementia** - Mild cognitive decline
4. **Moderate Dementia** - Moderate cognitive impairment

## Features

- 🧠 **Deep Learning Model**: VGG16 architecture fine-tuned on medical imaging data
- 🌐 **Web Interface**: User-friendly Flask application
- 📊 **Confidence Scores**: Probability distribution across all classes
- 🖼️ **Image Upload**: Support for common image formats (JPEG, PNG)
- ⚡ **Real-time Prediction**: Fast inference on CPU or GPU
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
alzymers/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── test_predictions.py             # Testing script for predictions
├── analyze_prediction_issue.py     # Debugging utilities
├── models/
│   └── vgg16_best_valacc.pth      # Trained model weights
├── resources/
│   ├── class_names.json           # Classification labels
│   └── Alzhemer's_VGG16_.ipynb    # Training notebook
├── static/
│   └── css/
│       └── styles.css             # Frontend styling
└── templates/
    ├── index.html                 # Home page
    ├── model.html                 # Model information page
    └── predict.html               # Prediction interface
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd alzymers
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files**
   Ensure the following files exist:
   - `models/vgg16_best_valacc.pth` - Trained model weights
   - `resources/class_names.json` - Class labels

## Usage

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. **Make predictions**
   - Click on "Get Started" or navigate to the prediction page
   - Upload an MRI brain scan image
   - Click "Predict" to get classification results
   - View the predicted class and confidence scores

### API Endpoints

The application provides the following endpoints:

- **`GET /`** - Home page
- **`GET /model`** - Model information and details
- **`GET /predict`** - Prediction interface
- **`POST /predict`** - API endpoint for image classification
  - Accepts: `multipart/form-data` with image file
  - Returns: JSON with prediction and probabilities

### Example API Usage

```python
import requests

url = 'http://localhost:5000/predict'
files = {'file': open('brain_scan.jpg', 'rb')}
response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Model Details

### Architecture

- **Base Model**: VGG16 pre-trained on ImageNet
- **Transfer Learning**: Fine-tuned on Alzheimer's MRI dataset
- **Custom Classifier**:
  - Fully Connected Layer: 25088 → 1024 (ReLU, BatchNorm, Dropout 0.5)
  - Fully Connected Layer: 1024 → 512 (ReLU, BatchNorm, Dropout 0.3)
  - Output Layer: 512 → 4 classes

### Training Configuration

- **Input Size**: 224×224 pixels
- **Normalization**: ImageNet mean and std
- **Frozen Layers**: First 20 convolutional layers
- **Trainable Layers**: Last convolutional blocks and custom classifier

### Image Preprocessing

Input images are automatically preprocessed:
1. Resized to 224×224 pixels
2. Converted to RGB format
3. Normalized using ImageNet statistics
4. Converted to PyTorch tensor

## Dependencies

- **Flask 3.0.0** - Web framework
- **PyTorch 2.1.0** - Deep learning framework
- **TorchVision 0.16.0** - Computer vision utilities
- **Pillow 10.1.0** - Image processing
- **NumPy 1.26.2** - Numerical computations
- **Gunicorn 21.2.0** - Production WSGI server
- **Werkzeug 3.0.1** - WSGI utilities

## Development

### Testing Predictions

Run the test script to verify model predictions:
```bash
python test_predictions.py
```

### Debugging

Use the analysis script to debug prediction issues:
```bash
python analyze_prediction_issue.py
```

## Deployment

### Production Deployment with Gunicorn

```bash
gunicorn --bind 0.0.0.0:8000 --workers 4 app:app
```

### Docker Deployment (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t alzheimers-classifier .
docker run -p 5000:5000 alzheimers-classifier
```

## Performance

- **Inference Time**: ~0.5-2 seconds per image (CPU)
- **Memory Usage**: ~500MB-1GB
- **Maximum Upload Size**: 8MB per image

## Limitations

- Model trained on specific MRI scan formats
- Best results with high-quality, properly oriented brain scans
- Not a replacement for professional medical diagnosis
- Should be used for research and educational purposes only

## Disclaimer

⚠️ **Medical Disclaimer**: This application is for educational and research purposes only. It should NOT be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

## Future Improvements

- [ ] Support for additional image formats (DICOM)
- [ ] Batch prediction capability
- [ ] Model performance metrics visualization
- [ ] User authentication and history tracking
- [ ] Integration with medical imaging standards
- [ ] Mobile application version

## License

This project is provided as-is for educational purposes.

## Acknowledgments

- VGG16 architecture by Visual Geometry Group, Oxford
- Pre-trained weights from PyTorch
- Alzheimer's MRI dataset contributors

## Contact

For questions or issues, please open an issue in the repository.

---

**Note**: Always ensure patient data privacy and comply with relevant healthcare regulations (HIPAA, GDPR, etc.) when working with medical imaging data.
