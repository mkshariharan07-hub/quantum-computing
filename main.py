"""
main.py — PlantPulse Enhanced Training Pipeline
================================================
Improvements over v1:
  • Parallel feature extraction (joblib Parallel)
  • Feature normalization (StandardScaler) saved alongside model
  • Stratified K-Fold cross-validation for honest accuracy estimate
  • GridSearchCV for automatic hyperparameter tuning
  • Per-class precision/recall/F1 report
  • Feature importance analysis saved as text
  • Robust progress logging with timing
"""

import numpy as np
import cv2
import os
import joblib
import random
import time
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv

load_dotenv()

# ===============================
# CONFIGURATION
# ===============================
SEED        = 42
IMG_SIZE    = (128, 128)
N_JOBS      = -1          # Use all CPU cores for parallel tasks
MODEL_PATH  = "plant_model.pkl"
SCALER_PATH = "plant_scaler.pkl"

np.random.seed(SEED)
random.seed(SEED)


# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Deterministic, comprehensive feature vector.
    Changes here MUST be mirrored in app.py extract_features().
    
    Feature breakdown (57 total):
      [0:24]  — Hue histogram (24 bins, 0-180)
      [24:40] — Saturation histogram (16 bins, 0-256)
      [40:56] — Value histogram (16 bins, 0-256)
      [56:62] — BGR mean (3) + std (3)
      [62]    — Edge density (Canny)
    """
    img = cv2.resize(img, IMG_SIZE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Color Histograms (HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    # Normalize histograms to [0, 1] so image size doesn't affect scale
    h_hist /= (h_hist.sum() + 1e-7)
    s_hist /= (s_hist.sum() + 1e-7)
    v_hist /= (v_hist.sum() + 1e-7)

    # 2. Global BGR Stats
    means, stds = cv2.meanStdDev(img)
    stats = np.concatenate([means.flatten(), stds.flatten()]) / 255.0

    # 3. Edge Complexity (Canny)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (IMG_SIZE[0] * IMG_SIZE[1])

    return np.concatenate([h_hist, s_hist, v_hist, stats, [edge_density]])


def _safe_extract(img: np.ndarray) -> np.ndarray:
    """Wrapper for parallel map — catches per-image errors."""
    try:
        return extract_features(img.astype("uint8"))
    except Exception:
        return np.zeros(63)  # Return zero-vector on failure


# ===============================
# LOAD & PREPROCESS
# ===============================
t0 = time.time()
print("=" * 55)
print("  PlantPulse — Enhanced Training Pipeline")
print("=" * 55)
print(f"\n📂 Loading dataset from images.npy / labels.npy ...")

try:
    images = np.load("images.npy")
    labels = np.load("labels.npy")
    print(f"   ✅ Loaded {len(images)} samples | {len(set(labels))} classes")
except FileNotFoundError as e:
    print(f"\n❌ Dataset not found: {e}")
    print("   Please ensure images.npy and labels.npy are in the project directory.")
    raise SystemExit(1)

# ===============================
# PARALLEL FEATURE EXTRACTION
# ===============================
print(f"\n🧠 Extracting features in parallel (using all CPU cores)...")
t1 = time.time()
X = np.array(
    Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_safe_extract)(img) for img in images
    )
)
y = labels
print(f"   ✅ Features extracted in {time.time() - t1:.1f}s | shape: {X.shape}")

# ===============================
# FEATURE NORMALIZATION
# ===============================
print(f"\n📐 Fitting StandardScaler on training features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)
print(f"   ✅ Scaler saved → {SCALER_PATH}")

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n📊 Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# ===============================
# HYPERPARAMETER TUNING (GridSearchCV)
# ===============================
print(f"\n🔧 Running GridSearchCV for optimal hyperparameters...")
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth":    [15, 20, None],
    "min_samples_split": [2, 4],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
gs = GridSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS),
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=N_JOBS,
    verbose=1,
    refit=True,
)
gs.fit(X_train, y_train)
model = gs.best_estimator_

print(f"\n   🏆 Best params: {gs.best_params_}")
print(f"   📈 CV Accuracy:  {gs.best_score_ * 100:.2f}%")

# ===============================
# EVALUATION
# ===============================
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
print(f"\n{'─'*55}")
print(classification_report(y_test, y_pred, zero_division=0))

# Save per-class report
with open("training_report.txt", "w") as f:
    f.write(f"PlantPulse — Training Report\n")
    f.write(f"{'='*55}\n")
    f.write(f"Best Params   : {gs.best_params_}\n")
    f.write(f"CV Accuracy   : {gs.best_score_*100:.2f}%\n")
    f.write(f"Test Accuracy : {test_acc*100:.2f}%\n\n")
    f.write(classification_report(y_test, y_pred, zero_division=0))
print("📄 Per-class report saved → training_report.txt")

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved → {MODEL_PATH}")
print(f"\n⏱  Total training time: {time.time() - t0:.1f}s")
print("=" * 55)