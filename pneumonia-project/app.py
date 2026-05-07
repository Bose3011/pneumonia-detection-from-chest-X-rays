from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
import traceback

app = Flask(__name__)

# -----------------------------
# 1. Load trained MobileNet model
# -----------------------------
# Use relative model paths so the app works when uploaded to GitHub
base_dir = os.path.dirname(os.path.abspath(__file__))
model_paths = [
    os.path.join(base_dir, "pneumonia_model_mobilenet.keras"),
    os.path.join(base_dir, "pneumonia_model.keras"),
    os.path.join(base_dir, "pneumonia_model_efficientnet.h5"),
]

model = None
class_names = None

for model_path in model_paths:
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)
            print(f"[OK] Model loaded successfully from: {model_path}")
            # Based on your dataset structure: BACTERIAL, COVID19, FUNGAL, NORMAL, VIRAL (alphabetical order)
            class_names = ["BACTERIAL", "COVID19", "FUNGAL", "NORMAL", "VIRAL"]
            break
    except Exception as e:
        print(f"[X] Failed to load {model_path}: {e}")
        continue

# Fallback dummy model if no model found
if model is None:
    print("[WARNING] No trained model found. Using dummy model for testing.")
    dummy_num_classes = 5
    model = Sequential([
        Flatten(input_shape=(224, 224, 3)),
        Dense(128, activation='relu'),
        Dense(dummy_num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    class_names = ["BACTERIAL", "COVID19", "FUNGAL", "NORMAL", "VIRAL"]

print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
print(f"Classes: {class_names}")


# -----------------------------
# 2. Mapping to pneumonia type + inference text
# -----------------------------
inference_map = {
    "NORMAL": {
        "type": "None",
        "text": "No clear radiographic signs of pneumonia are seen in this X-ray image. The lungs appear normal."
    },
    "BACTERIAL": {
        "type": "Bacterial Pneumonia",
        "text": "The pattern suggests consolidation that is more typical of bacterial pneumonia. "
                "Clinical correlation, blood tests, and sputum culture are recommended for confirmation."
    },
    "VIRAL": {
        "type": "Viral Pneumonia",
        "text": "Diffuse or interstitial changes may be consistent with viral pneumonia. "
                "Correlation with symptoms and viral PCR/antigen tests is advised."
    },
    "COVID19": {
        "type": "COVID-19 Pneumonia",
        "text": "Radiographic features are consistent with COVID-19 pneumonia. "
                "PCR testing and clinical correlation are essential for diagnosis."
    },
    "FUNGAL": {
        "type": "Fungal Pneumonia",
        "text": "Patterns suggest possible fungal pneumonia. "
                "Fungal cultures and serological tests are recommended, especially in immunocompromised patients."
    },
}


@app.route("/")
def home():
    """Render the dark-themed upload interface."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Handle GET requests (user navigated directly to /predict)
    if request.method == "GET":
        return render_template("error.html", error="Please use the upload form on the home page to submit an image."), 400
    """
    Receive an image from the interface, run it through the classifier,
    and return: whether pneumonia is present, the pneumonia type, and inference.
    """
    try:
        # Debug: Print request info
        print(f"\n=== PREDICTION REQUEST ===")
        print(f"Request method: {request.method}")
        print(f"Content type: {request.content_type}")
        print(f"Request files keys: {list(request.files.keys())}")
        print(f"Form keys: {list(request.form.keys())}")
        
        # Check if file is in request
        if "file" not in request.files:
            error_msg = f"No 'file' key in request.files. Available keys: {list(request.files.keys())}"
            print(f"ERROR: {error_msg}")
            return render_template("error.html", error=error_msg), 400

        file = request.files["file"]
        print(f"File received: {file.filename}")
        
        if file.filename == "":
            return render_template("error.html", error="No file selected. Please choose an image file."), 400

        # Basic extension check
        allowed_extensions = {"png", "jpg", "jpeg", "gif"}
        file_ext = file.filename.rsplit(".", 1)[1].lower() if "." in file.filename else ""
        if file_ext not in allowed_extensions:
            return render_template("error.html", error=f"Invalid file type '{file_ext}'. Allowed: {allowed_extensions}"), 400

        # Save uploaded image
        os.makedirs("static", exist_ok=True)
        img_path = os.path.join("static", "uploaded_image.jpg")
        file.save(img_path)
        print(f"Image saved to: {img_path}")

        # Preprocess image (224x224 RGB, 0-1 scaled)
        img = load_img(img_path, target_size=(224, 224), color_mode="rgb")
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        print(f"Image shape: {img_array.shape}")

        # Multi-class softmax prediction
        preds = model.predict(img_array, verbose=0)[0]  # shape: (num_classes,)
        print(f"Raw predictions: {preds}")
        print(f"Prediction shape: {preds.shape}")
        
        predicted_idx = int(np.argmax(preds))
        predicted_class = class_names[predicted_idx]
        confidence = float(preds[predicted_idx])
        
        print(f"Predicted class: {predicted_class} (index {predicted_idx})")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

        # Determine pneumonia presence and type
        info = inference_map.get(
            predicted_class,
            {
                "type": "Unknown",
                "text": "No specific inference available for this class.",
            },
        )

        has_pneumonia = predicted_class != "NORMAL"
        pneumonia_type = info["type"]
        inference_text = info["text"]

        # Determine severity based on confidence score
        if has_pneumonia:
            if confidence >= 0.8:
                severity = "High"
            elif confidence >= 0.5:
                severity = "Moderate"
            else:
                severity = "Low"
        else:
            severity = "N/A"  # No severity for normal cases

        print(f"Has pneumonia: {has_pneumonia}")
        print(f"Pneumonia type: {pneumonia_type}")
        print(f"Severity: {severity}")
        print("=" * 30 + "\n")

        return render_template(
            "result.html",
            result=predicted_class,
            has_pneumonia=has_pneumonia,
            pneumonia_type=pneumonia_type,
            severity=severity,
            confidence=confidence,
            inference=inference_text,
            img_path=url_for("static", filename="uploaded_image.jpg"),
        )
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"ERROR in predict(): {e}")
        print(error_trace)
        return render_template("error.html", error=f"Prediction error: {str(e)}"), 500


@app.route("/test", methods=["GET"])
def test():
    """Test route to verify the app is running."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "classes": class_names,
        "model_input_shape": str(model.input_shape) if model else None,
    })


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Flask app...")
    print("="*50)
    app.run(debug=True, host="127.0.0.1", port=5000)