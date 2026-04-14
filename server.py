from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def load_model():
    global model
    try:
        if os.path.exists("plant_model.pkl"):
            model = joblib.load("plant_model.pkl")
            logger.info("✅ Model loaded successfully.")
        else:
            logger.warning("⚠️ Model file 'plant_model.pkl' not found. API will return errors until model is trained.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

load_model()

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided in the request."}), 400

    file = request.files["image"]
    
    try:
        # Read image
        img_buffer = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format or corrupted file."}), 400

        # Preprocess (128x128 color)
        img_resized = cv2.resize(img, (128, 128))
        features = img_resized.flatten().reshape(1, -1)

        # check dimensions
        if features.shape[1] != model.n_features_in_:
            return jsonify({
                "error": "Model dimension mismatch",
                "expected": model.n_features_in_,
                "got": features.shape[1]
            }), 400

        # AI Prediction
        prediction = model.predict(features)[0]
        conf_probs = model.predict_proba(features)
        confidence = float(np.max(conf_probs) * 100)

        # Split prediction string safely
        if "___" in prediction:
            plant, disease = prediction.split("___")
        else:
            plant, disease = "Unknown", prediction

        return jsonify({
            "status": "success",
            "plant": plant,
            "disease": disease,
            "confidence": confidence,
            "prediction_string": prediction
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "api": "v1"
    })

@app.route("/")
def home():
    return "✅ Plant Disease API is running. Use /predict (POST) or /health (GET)."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False) # Turn off debug for stability