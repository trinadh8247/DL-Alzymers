# Alzhemer's VGG16 - Modularized

This repository contains a modular refactor of the original notebook along with a small Flask app for serving predictions.

Structure
- src/: modular python package (data, model, train, predict, utils)
- app.py: Flask backend for inference
- templates/index.html: small upload UI
- requirements.txt: Python dependencies

Quick start (Windows / PowerShell)

1. Create a virtual environment and install dependencies

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Run the Flask app

```powershell
python app.py
```

3. Open http://localhost:5000 and upload an image to get predictions.

Notes
- The Flask app expects a checkpoint named `vgg16_best_valacc.pth` in the project root. If missing, the model will be a freshly initialized VGG16.
- The training functions are in `src/train.py`. Use the modular notebook to run training locally or adapt to your cluster.
