import numpy as np
import cv2
import os
import joblib
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

load_dotenv()

# ===============================
# CONFIGURATION
# ===============================
np.random.seed(42)
random.seed(42)
IMG_SIZE = (128, 128)

def extract_features(img):
    """
    Enhanced Feature Extraction:
    - Color Histograms (HSV)
    - Statistical Moments (Mean/Std)
    """
    img = cv2.resize(img, IMG_SIZE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. Color Histograms (Finer Detail)
    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    
    # 2. Global Stats
    means, stds = cv2.meanStdDev(img)
    stats = np.concatenate([means.flatten(), stds.flatten()])

    # 3. Edge Complexity (Crucial for Tomato vs Potato shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (IMG_SIZE[0] * IMG_SIZE[1])
    
    return np.concatenate([h_hist, s_hist, v_hist, stats, [edge_density]])

# ===============================
# LOAD & PREPROCESS
# ===============================
print("📂 Loading dataset...")
try:
    images = np.load("images.npy")
    labels = np.load("labels.npy")
    
    print("🧠 Extracting advanced features (Color Histograms + Stats)...")
    X = []
    for img in images:
        X.append(extract_features(img.astype("uint8")))
    X = np.array(X)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ===============================
    # TRAIN MODEL
    # ===============================
    print("🚀 Training Enhanced Random Forest...")
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Training Complete. Accuracy: {acc*100:.2f}%")

    joblib.dump(model, "plant_model.pkl")
    print("💾 Model saved as plant_model.pkl")

except Exception as e:
    print(f"❌ Error during training: {e}")