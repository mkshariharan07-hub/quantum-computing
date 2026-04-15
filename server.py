"""
server.py — PlantPulse Flask REST API  (Enhanced v2)
=====================================================
Fixes vs v1:
  • Uses correct extract_features() — NOT raw img.flatten()
  • Loads StandardScaler and applies it before prediction
  • /classes endpoint to list all known disease classes
  • /metrics endpoint for Prometheus-style monitoring
  • Request size guard (5 MB max)
  • Structured JSON error responses with "code" field
  • Logging includes confidence and plant name per request
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import joblib
import os
import logging
import time

# ── Configure Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MAX_CONTENT_LENGTH = 5 * 1024 * 1024   # 5 MB upload limit
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ── Global State ───────────────────────────────────────────────────────────────
model  = None
scaler = None
REQUEST_COUNT  = 0
ERROR_COUNT    = 0
TOTAL_LATENCY  = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (must mirror main.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════
def extract_features(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    h_hist /= (h_hist.sum() + 1e-7)
    s_hist /= (s_hist.sum() + 1e-7)
    v_hist /= (v_hist.sum() + 1e-7)

    means, stds = cv2.meanStdDev(img)
    stats = np.concatenate([means.flatten(), stds.flatten()]) / 255.0

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (128 * 128)

    return np.concatenate([h_hist, s_hist, v_hist, stats, [edge_density]])


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP — Load Model & Scaler
# ═══════════════════════════════════════════════════════════════════════════════
def load_artifacts():
    global model, scaler
    try:
        if os.path.exists("plant_model.pkl"):
            model = joblib.load("plant_model.pkl")
            logger.info(f"✅ Model loaded — {len(model.classes_)} classes, "
                        f"{model.n_features_in_} features.")
        else:
            logger.warning("⚠️  plant_model.pkl not found — run python main.py first.")

        if os.path.exists("plant_scaler.pkl"):
            scaler = joblib.load("plant_scaler.pkl")
            logger.info("✅ Scaler loaded.")
        else:
            logger.warning("⚠️  plant_scaler.pkl not found — predictions may be less accurate.")
    except Exception as e:
        logger.error(f"❌ Failed to load artifacts: {e}")

load_artifacts()


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return jsonify({
        "service": "PlantPulse API",
        "version": "2.0",
        "endpoints": ["/predict", "/health", "/classes", "/metrics"],
        "model_loaded": model is not None,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "api_version": "v2",
    })


@app.route("/classes", methods=["GET"])
def classes():
    if model is None:
        return jsonify({"code": "MODEL_NOT_LOADED", "error": "Model not loaded."}), 503
    return jsonify({
        "count": len(model.classes_),
        "classes": list(model.classes_),
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    avg_lat = (TOTAL_LATENCY / REQUEST_COUNT) if REQUEST_COUNT > 0 else 0
    return jsonify({
        "total_requests": REQUEST_COUNT,
        "total_errors": ERROR_COUNT,
        "avg_latency_ms": round(avg_lat * 1000, 2),
    })


@app.route("/predict", methods=["POST"])
def predict():
    global REQUEST_COUNT, ERROR_COUNT, TOTAL_LATENCY
    REQUEST_COUNT += 1
    t_start = time.perf_counter()

    if model is None:
        ERROR_COUNT += 1
        return jsonify({"code": "MODEL_NOT_LOADED",
                        "error": "Model not available. Train with python main.py."}), 503

    if "image" not in request.files:
        ERROR_COUNT += 1
        return jsonify({"code": "NO_IMAGE", "error": "No image file in request."}), 400

    file = request.files["image"]

    try:
        # Decode image
        buf = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            ERROR_COUNT += 1
            return jsonify({"code": "INVALID_IMAGE",
                            "error": "Cannot decode image. Ensure it is a valid JPG/PNG."}), 400

        # Extract features (correct — not raw pixels)
        raw_features = extract_features(img).reshape(1, -1)

        if raw_features.shape[1] != model.n_features_in_:
            ERROR_COUNT += 1
            return jsonify({
                "code": "FEATURE_MISMATCH",
                "error": "Feature dimension mismatch.",
                "expected": int(model.n_features_in_),
                "got": int(raw_features.shape[1]),
            }), 400

        # Apply scaler if available
        features = scaler.transform(raw_features) if scaler else raw_features

        # Predict
        prediction  = model.predict(features)[0]
        conf_probs  = model.predict_proba(features)[0]
        confidence  = float(np.max(conf_probs) * 100)

        plant, disease = prediction.split("___") if "___" in prediction else ("Unknown", prediction)

        # Top-3 alternatives
        top3 = sorted(
            [{"class": c, "probability": round(float(p) * 100, 2)}
             for c, p in zip(model.classes_, conf_probs)],
            key=lambda x: -x["probability"]
        )[:3]

        latency = time.perf_counter() - t_start
        TOTAL_LATENCY += latency
        logger.info(f"Predicted: {plant} / {disease} | conf={confidence:.1f}% | {latency*1000:.1f}ms")

        return jsonify({
            "status": "success",
            "plant": plant,
            "disease": disease,
            "confidence_pct": round(confidence, 2),
            "top3": top3,
            "latency_ms": round(latency * 1000, 2),
            "scaler_applied": scaler is not None,
        })

    except Exception as e:
        ERROR_COUNT += 1
        logger.error(f"Prediction error: {e}")
        return jsonify({"code": "SERVER_ERROR", "error": str(e)}), 500


# ── 413 handler for oversized uploads ─────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    return jsonify({"code": "FILE_TOO_LARGE",
                    "error": f"Image exceeds 5 MB limit."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)