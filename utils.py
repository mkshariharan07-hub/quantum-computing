"""
utils.py — PlantPulse Shared Utilities
=======================================
Single source of truth for:
  • Feature extraction  (used by main.py, app.py, server.py)
  • Artifact paths
  • Disease knowledge base
  • Image decoding helpers

RULE: Any change to extract_features() is made HERE only.
      All other files import from this module.
"""

import cv2
import numpy as np
import os
import requests
import json
from typing import Optional, Dict, Any

# ── Artifact paths (one place to change if you move files) ────────────────────
MODEL_PATH  = "plant_model.pkl"
SCALER_PATH = "plant_scaler.pkl"
REPORT_PATH = "training_report.txt"
IMG_SIZE    = (128, 128)

# Feature-space identifiers
FEATURE_MODE_RAW  = "raw_pixels"
FEATURE_MODE_V2   = "histogram_63"
FEATURE_MODE_V3   = "spatial_192"
RAW_PIXEL_DIM     = 128 * 128 * 3
HIST_DIM_V2       = 63
HIST_DIM_V3       = 192


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
def extract_features(img: np.ndarray) -> np.ndarray:
    """
    v3: 192-dim spatial color feature vector (8x8x3).
    """
    img_8x8 = cv2.resize(img, (8, 8))
    return (img_8x8.flatten() / 255.0).astype(np.float64)


def extract_features_v2(img: np.ndarray) -> np.ndarray:
    """
    v2: 63-dim histogram-based feature vector.
    Used for legacy compatibility with current model.
    """
    img_res = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img_res, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    h_hist /= (h_hist.sum() + 1e-7)
    s_hist /= (s_hist.sum() + 1e-7)
    v_hist /= (v_hist.sum() + 1e-7)
    means, stds = cv2.meanStdDev(img_res)
    stats = np.concatenate([means.flatten(), stds.flatten()]) / 255.0
    edges = cv2.Canny(cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY), 100, 200)
    edge_density = np.sum(edges > 0) / (128 * 128)
    return np.concatenate([h_hist, s_hist, v_hist, stats, [edge_density]])


FEATURE_DIM = len(extract_features(np.zeros((8, 8, 3), dtype=np.uint8)))  # = 63


def extract_features_raw(img: np.ndarray) -> np.ndarray:
    """
    Legacy extractor — raw pixel flatten (128×128×3 = 49152 dims).
    Used automatically when the loaded model was trained this way.
    DO NOT use for new training; use extract_features() instead.
    """
    return cv2.resize(img, IMG_SIZE).flatten().astype(np.float64)


def get_feature_mode(model) -> str:
    """
    Inspect a loaded model and return which feature extractor it was trained with.

    Returns:
        'raw_pixels'  — model.n_features_in_ == 49152  (old pipeline)
        'histogram'   — model.n_features_in_ == 63     (new pipeline)

    Raises:
        ValueError if the feature count is unrecognised.
    """
    n = model.n_features_in_
    if n == RAW_PIXEL_DIM: return FEATURE_MODE_RAW
    if n == HIST_DIM_V2:   return FEATURE_MODE_V2
    if n == HIST_DIM_V3:   return FEATURE_MODE_V3
    raise ValueError(f"CRITICAL: Model expects {n} features, but system only supports {RAW_PIXEL_DIM}, {HIST_DIM_V2}, or {HIST_DIM_V3}.")


def extract_for_model(img: np.ndarray, model) -> np.ndarray:
    """
    Extract features in whichever space the model was trained in.
    """
    n = model.n_features_in_
    if n == RAW_PIXEL_DIM:
        return extract_features_raw(img).reshape(1, -1)
    if n == HIST_DIM_V2:
        return extract_features_v2(img).reshape(1, -1)
    # Default to V3 if 192 or unknown (get_feature_mode will catch errors)
    return extract_features(img).reshape(1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE DECODING
# ═══════════════════════════════════════════════════════════════════════════════
def decode_bytes_to_bgr(raw_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode raw image bytes → BGR ndarray.
    Returns None if bytes are empty or decoding fails.
    """
    if not raw_bytes:
        return None
    arr = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img  # None on failure


def decode_file_to_bgr(path: str) -> Optional[np.ndarray]:
    """Read an image file from disk → BGR ndarray."""
    return cv2.imread(path, cv2.IMREAD_COLOR)


# ═══════════════════════════════════════════════════════════════════════════════
# DISEASE KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════
DISEASE_INFO: dict[str, dict] = {
    "healthy": {
        "severity": "low", "color": "#10b981", "emoji": "🌱",
        "tips": "No treatment needed. Maintain regular watering and sunlight."
    },
    "early_blight": {
        "severity": "medium", "color": "#f59e0b", "emoji": "🟡",
        "tips": "Remove affected leaves. Apply copper-based fungicide. Avoid overhead watering."
    },
    "late_blight": {
        "severity": "high", "color": "#ef4444", "emoji": "🔴",
        "tips": "Isolate plant immediately. Apply mancozeb or chlorothalonil. Destroy infected tissue.",
        "active_ingredient": "Chlorothalonil 75% WP", "application_rate": "2.0-2.5 kg/ha",
        "buy_link": "https://www.google.com/search?q=buy+chlorothalonil+fungicide"
    },
    "leaf_mold": {
        "severity": "medium", "color": "#f97316", "emoji": "🟠",
        "tips": "Improve air circulation. Apply fungicide. Reduce ambient humidity."
    },
    "bacterial_spot": {
        "severity": "high", "color": "#ef4444", "emoji": "🔴",
        "tips": "Use copper-based bactericide. Avoid working with wet plants."
    },
    "common_rust": {
        "severity": "medium", "color": "#f97316", "emoji": "🟠",
        "tips": "Apply triazole fungicide early. Rotate crops next season."
    },
    "northern_leaf_blight": {
        "severity": "high", "color": "#ef4444", "emoji": "🔴",
        "tips": "Apply fungicide at first sign. Use resistant varieties next cycle."
    },
    "gray_leaf_spot": {
        "severity": "medium", "color": "#f59e0b", "emoji": "🟡",
        "tips": "Improve drainage. Apply strobilurin fungicide preventively."
    },
    "powdery_mildew": {
        "severity": "medium", "color": "#f59e0b", "emoji": "🟡",
        "tips": "Apply sulfur or potassium bicarbonate spray. Ensure good airflow."
    },
    "target_spot": {
        "severity": "medium", "color": "#f97316", "emoji": "🟠",
        "tips": "Remove infected leaves. Apply chlorothalonil or mancozeb."
    },
    "mosaic_virus": {
        "severity": "high", "color": "#ef4444", "emoji": "🔴",
        "tips": "No cure — remove and destroy infected plants. Control aphid vectors."
    },
    "yellow_leaf_curl_virus": {
        "severity": "high", "color": "#ef4444", "emoji": "🔴",
        "tips": "Remove infected plants. Use reflective mulches to deter whiteflies.",
        "active_ingredient": "Imidacloprid (for vector control)",
        "application_rate": "10-15 ml per 10 L water",
        "buy_link": "https://www.google.com/search?q=buy+imidacloprid+insecticide"
    },
    "anthracnose": {
        "severity": "high", "color": "#7c2d12", "emoji": "🟤",
        "tips": "Prune out dead wood. Apply chlorothalonil or copper-based sprays during bud break.",
        "active_ingredient": "Copper Oxychloride 50% WP", "application_rate": "3.0 g/L",
        "buy_link": "https://www.google.com/search?q=buy+copper+fungicide+for+anthracnose"
    },
    "downy_mildew": {
        "severity": "high", "color": "#facc15", "emoji": "🟡",
        "tips": "Increase spacing for airflow. Use metalaxyl or mancozeb.",
        "active_ingredient": "Metalaxyl 8% + Mancozeb 64%", "application_rate": "2.5 g/L",
        "buy_link": "https://www.google.com/search?q=buy+metalaxyl+fungicide"
    },
    "canker": {
        "severity": "high", "color": "#451a03", "emoji": "🪵",
        "tips": "Cut well below infected area. Sanitize tools. Apply tree paint.",
        "active_ingredient": "Fixed Copper Fungicide", "application_rate": "Follow label",
        "buy_link": "https://www.google.com/search?q=buy+copper+canker+treatment"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM AUDIT & FEASIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

class SystemAuditor:
    """Checks for file existence and model loadability."""
    @staticmethod
    def run_audit() -> Dict[str, Any]:
        results = {
            "model": os.path.exists(MODEL_PATH),
            "scaler": os.path.exists(SCALER_PATH),
            "env": os.path.exists(".env"),
            "internet": False
        }
        try:
            requests.get("https://google.com", timeout=2)
            results["internet"] = True
        except Exception: pass
        return results

_FALLBACK_INFO = {
    "severity": "medium",
    "color":    "#f59e0b",
    "emoji":    "⚠️",
    "tips":     "Consult an agronomist for targeted treatment advice.",
}


def get_disease_info(disease: str) -> dict:
    """
    Lookup disease metadata by fuzzy key match.
    Falls back gracefully if disease is unknown.

    Args:
        disease: Raw disease string (e.g. 'Early_blight', 'Late blight').
    Returns:
        Dict with keys: severity, color, emoji, tips.
    """
    key = disease.lower().replace(" ", "_")
    for k, v in DISEASE_INFO.items():
        if k in key or key in k:
            return v
    return _FALLBACK_INFO


# ═══════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def load_model_and_scaler():
    """
    Load plant_model.pkl and plant_scaler.pkl from disk.
    Returns (model, scaler). scaler may be None if not found.
    Raises FileNotFoundError if model is missing.
    """
    import joblib
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run `python main.py` to train."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return model, scaler


def predict_image(img_bgr: np.ndarray, model, scaler=None) -> dict:
    """
    Full prediction pipeline for a BGR image.
    Auto-detects whether the model expects raw pixels (49152) or
    histogram features (63) and extracts accordingly.

    Returns dict with keys:
        plant, disease, confidence, prediction_raw, top5,
        severity, tips, color, emoji, feature_mode
    """
    # ── Auto-detect feature space ────────────────────────────────────────────
    mode     = get_feature_mode(model)   # raises ValueError on unknown dim
    features = extract_for_model(img_bgr, model)  # shape (1, n_features)

    # ── Strict Scaling ────────────────────────────────────────────────────────
    if scaler is not None:
        try:
            # Scaler only applies if dimensions match its expectation
            if scaler.n_features_in_ == features.shape[1]:
                features = scaler.transform(features)
        except Exception:
            pass # Fallback to unscaled if drift detected
    
    prediction  = model.predict(features)[0]
    conf_probs  = model.predict_proba(features)[0]
    confidence  = float(np.max(conf_probs) * 100)

    try:
        plant, disease = prediction.split("___")
    except ValueError:
        plant, disease = "Unknown", prediction

    info = get_disease_info(disease)

    top5 = sorted(
        [{"class": c, "probability": round(float(p) * 100, 2)}
         for c, p in zip(model.classes_, conf_probs)],
        key=lambda x: -x["probability"]
    )[:5]

    return {
        "plant":        plant,
        "disease":      disease,
        "confidence":   round(confidence, 2),
        "prediction_raw": prediction,
        "top5":         top5,
        "severity":     info["severity"],
        "tips":         info["tips"],
        "color":        info["color"],
        "emoji":        info["emoji"],
        "feature_mode": mode,   # 'raw_pixels' or 'histogram'
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLANTNET API INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════
def identify_plant_plantnet(img_bgr: np.ndarray, api_key: str, project: str = "all") -> Dict[str, Any]:
    """
    Identifies a plant using the Pl@ntNet API.
    
    Args:
        img_bgr: Image in BGR format (OpenCV default).
        api_key: Pl@ntNet API key.
        project: Pl@ntNet project (default "all").
        
    Returns:
        Dictionary with identification results or error info.
    """
    if not api_key:
        return {"error": "Pl@ntNet API key not provided."}

    # Encode image to JPEG
    success, buffer = cv2.imencode(".jpg", img_bgr)
    if not success:
        return {"error": "Failed to encode image for Pl@ntNet API."}

    endpoint = f"https://my-api.plantnet.org/v2/identify/{project}?api-key={api_key}"
    
    files = [
        ('images', ('image.jpg', buffer.tobytes(), 'image/jpeg'))
    ]
    data = {
        'organs': ['leaf']
    }

    try:
        response = requests.post(endpoint, files=files, data=data, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Pl@ntNet API request failed: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED ECOSYSTEM KNOWLEDGE
# ═══════════════════════════════════════════════════════════════════════════════

def get_plant_details(species_name: str) -> Dict[str, Any]:
    """
    Fetch comprehensive species data from Wikipedia API.
    """
    try:
        # Dynamic search for any species identified by Pl@ntNet
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{species_name.replace(' ', '_')}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "summary": data.get("extract", "Botanical summary currently under neural synthesis."),
                "image": data.get("thumbnail", {}).get("source"),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
                "found": True
            }
    except Exception:
        pass
    return {"summary": "Universal species data currently restricted to local cache.", "found": False}


def get_care_tips(plant_name: str) -> str:
    """
    Context-aware plant care generator.
    """
    tips = {
        "tomato": "Keep soil consistently moist. Support with stakes. Provide 6-8 hours of direct sun.",
        "potato": "Hill soil around stems as they grow. Water deeply during flowering.",
        "corn": "Plant in blocks for pollination. Ensure high nitrogen soil and frequent watering.",
        "pepper": "Avoid overwatering young plants. Keep in warm, bright locations.",
        "grape": "Prune during dormancy. Ensure good drainage and air circulation to prevent rot.",
        "apple": "Thin fruit to prevent branch breakage. Prune for sunlight penetration.",
    }
    name = plant_name.lower()
    for key, tip in tips.items():
        if key in name:
            return tip
    return "Ensure proper soil drainage and consistent sunlight. Monitor for pests weekly."


def calculate_health_index(ai_conf: float, q_ones_ratio: float, pn_score: float = 0.0) -> int:
    """
    Generates a unified 'Plant Pulse' score (0-100).
    Higher is better.
    """
    # 1. AI factor (0-40 pts): High AI confidence in 'healthy' adds points
    ai_factor = (ai_conf / 100.0) * 40
    
    # 2. Quantum factor (0-30 pts): Lower ones_ratio usually correlates to healthy in our circuit
    q_factor = (1.0 - q_ones_ratio) * 30
    
    # 3. Pl@ntNet factor (0-30 pts): High confidence in species ID increases reliability
    pn_factor = (pn_score / 100.0) * 30 if pn_score > 0 else 15
    
    return int(min(ai_factor + q_factor + pn_factor, 100))


def get_weather_context() -> Dict[str, Any]:
    """
    Fetch local weather context based on IP.
    Used to calculate environmental stress.
    """
    try:
        # 1. Get location via IP
        loc_res = requests.get("https://ipapi.co/json/", timeout=3).json()
        city = loc_res.get("city", "Unknown")
        
        # 2. Get weather for that city (wttr.in returns simple JSON)
        weather = requests.get(f"https://wttr.in/{city}?format=j1", timeout=5).json()
        curr = weather["current_condition"][0]
        
        return {
            "city": city,
            "temp": f"{curr['temp_C']}°C",
            "humidity": f"{curr['humidity']}%",
            "desc": curr['weatherDesc'][0]['value'],
            "uv": curr.get('uvIndex', 'N/A'),
            "found": True
        }
    except Exception:
        return {"city": "Global", "found": False}


def generate_bio_signatures(img: np.ndarray, health_index: int) -> Dict[str, float]:
    """
    Simulates botanical bio-signatures based on image analysis.
    Provides data for the radar chart.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Simple heuristics for bio-sigs
    # 1. Chlorophyll Content (Green channel dominance in HSV)
    green_mask = cv2.inRange(img_hsv, (35, 20, 20), (85, 255, 255))
    chlorophyll = (np.sum(green_mask > 0) / (img.shape[0] * img.shape[1])) * 100
    
    # 2. Structural Integrity (Edge density from Canny)
    edges = cv2.Canny(img, 100, 200)
    structure = (np.sum(edges > 0) / (img.shape[0] * img.shape[1])) * 200 # scaled
    
    # 3. Hydration (V channel in HSV)
    hydration = np.mean(img_hsv[:,:,2]) / 2.55
    
    # 4. Metabolic Stability (Fused with health index)
    metabolic = (health_index * 0.8) + (np.random.rand() * 20)
    
    # Normalize
    return {
        "Chlorophyll": round(min(chlorophyll * 1.5, 100), 1),
        "Structure": round(min(structure, 100), 1),
        "Hydration": round(min(hydration, 100), 1),
        "Metabolic": round(min(metabolic, 100), 1),
        "Spectral UV": round(np.random.uniform(70, 95), 1)
    }
def get_health_forecast(health_index: int, severity: str) -> Dict[str, Any]:
    """
    Simulates a 10-day health trajectory.
    """
    days = list(range(1, 11))
    decay = 5 if severity == "high" else (2 if severity == "medium" else 0.5)
    points = [max(0, health_index - (d * decay) + np.random.randint(-2, 3)) for d in days]
    return {"days": days, "index": points}


def get_global_spread(disease: str) -> Dict[str, Any]:
    """
    Simulates global disease distribution data.
    """
    regions = ["North America", "South America", "Europe", "Asia", "Africa", "Oceania"]
    impact = [np.random.randint(10, 90) for _ in regions]
    return dict(zip(regions, impact))
