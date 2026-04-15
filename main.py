"""
main.py — PlantPulse Training Pipeline v2
==========================================
Uses shared utils.py for extract_features() — no duplication.
"""

import numpy as np
import os
import joblib
import random
import time
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv

from utils import extract_features, MODEL_PATH, SCALER_PATH, REPORT_PATH, FEATURE_DIM

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
SEED   = 42
N_JOBS = -1
np.random.seed(SEED)
random.seed(SEED)


def _safe_extract(img: np.ndarray) -> np.ndarray:
    try:
        return extract_features(img.astype("uint8"))
    except Exception:
        return np.zeros(FEATURE_DIM)


# ── Load Dataset ───────────────────────────────────────────────────────────────
t0 = time.time()
print("=" * 55)
print("  PlantPulse — Training Pipeline v2")
print("=" * 55)

try:
    images = np.load("images.npy")
    labels = np.load("labels.npy")
    print(f"\n📂 Loaded {len(images)} samples | {len(set(labels))} classes")
except FileNotFoundError as e:
    print(f"\n❌ Dataset not found: {e}")
    print("   Ensure images.npy and labels.npy are in the project directory.")
    raise SystemExit(1)

# ── Parallel Feature Extraction ───────────────────────────────────────────────
print(f"\n🧠 Extracting features in parallel...")
t1 = time.time()
X = np.array(
    Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_safe_extract)(img) for img in images
    )
)
y = labels
print(f"   ✅ Done in {time.time()-t1:.1f}s | shape: {X.shape}")

# ── Fit & Save Scaler ─────────────────────────────────────────────────────────
print(f"\n📐 Fitting StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)
print(f"   ✅ Scaler saved → {SCALER_PATH}")

# ── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# ── GridSearchCV ──────────────────────────────────────────────────────────────
print(f"\n🔧 GridSearchCV (this may take a few minutes)...")
param_grid = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [15, 20, None],
    "min_samples_split": [2, 4],
}
gs = GridSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS),
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring="accuracy",
    n_jobs=N_JOBS,
    verbose=1,
    refit=True,
)
gs.fit(X_train, y_train)
model = gs.best_estimator_
print(f"\n   🏆 Best params : {gs.best_params_}")
print(f"   📈 CV Accuracy  : {gs.best_score_*100:.2f}%")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred   = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
report   = classification_report(y_test, y_pred, zero_division=0)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%\n")
print(report)

with open(REPORT_PATH, "w") as f:
    f.write(f"PlantPulse Training Report\n{'='*55}\n")
    f.write(f"Best Params   : {gs.best_params_}\n")
    f.write(f"CV Accuracy   : {gs.best_score_*100:.2f}%\n")
    f.write(f"Test Accuracy : {test_acc*100:.2f}%\n\n")
    f.write(report)

# ── Save Model ────────────────────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved → {MODEL_PATH}")
print(f"📄 Report saved → {REPORT_PATH}")
print(f"\n⏱  Total time: {time.time()-t0:.1f}s")
print("=" * 55)