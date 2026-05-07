# Pneumonia X-Ray Classifier Web App

This repository contains a minimal Flask web application for pneumonia classification from chest X-ray images.

## What is included
- `app.py` - Flask web app that loads a trained model and runs image predictions
- `pneumonia_model_mobilenet.keras` - the trained model file used by the app
- `templates/` - HTML templates for the web interface
- `static/` - static assets and uploaded image storage
- `dataset/` - image dataset folders (`train/`, `test/`, `val/`)

## Run locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   python app.py
   ```
3. Open the browser at `http://127.0.0.1:5000`

## Notes
- The app looks for the model file in the repository root.
- Uploaded images are stored temporarily in `static/uploaded_image.jpg`.
- The dataset folders are preserved in `dataset/` for reuse or training.
