from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("plant_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))

    features = img.flatten().reshape(1, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    prediction = model.predict(features)[0]
    confidence = float(np.max(model.predict_proba(features)) * 100)

    plant, disease = prediction.split("___")

    return jsonify({
        "plant": plant,
        "disease": disease,
        "confidence": confidence
    })
@app.route("/")
def home():
    return "✅ Plant Disease API is running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)