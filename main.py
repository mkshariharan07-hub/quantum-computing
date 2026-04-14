import numpy as np
import cv2
import os
import joblib
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ===============================
# SET RANDOM SEED (STABILITY)
# ===============================

np.random.seed(42)
random.seed(42)

# ===============================
# LOAD DATASET
# ===============================

images = np.load("images.npy")
labels = np.load("labels.npy")

print("Dataset loaded")
print("Total images:", len(images))

# ===============================
# PREPROCESS DATASET (IMPROVED)
# ===============================

processed_images = []

for img in images:
    img = img.astype("uint8")
    img = cv2.resize(img, (128, 128))   # improved size
    processed_images.append(img.flatten())  # keep color

processed_images = np.array(processed_images)

print("Dataset preprocessed")

# ===============================
# TRAIN / LOAD MODEL
# ===============================

if not os.path.exists("plant_model.pkl"):
    print("\nTraining model...")

    X_train, X_test, y_train, y_test = train_test_split(
        processed_images, labels, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "plant_model.pkl")
    print("Model trained and saved")

else:
    model = joblib.load("plant_model.pkl")
    print("Model loaded")

# ===============================
# USER INPUT IMAGE
# ===============================

print("\n==============================")
print("PLANT DISEASE DETECTION (AI + QUANTUM)")
print("==============================")

image_path = input("Enter image path: ")

if not os.path.exists(image_path):
    print("❌ Image not found")
    exit()

img = cv2.imread(image_path)

if img is None:
    print("❌ Unable to read image")
    exit()

# ===============================
# PREPROCESS INPUT IMAGE
# ===============================

img = cv2.resize(img, (128, 128))
features = img.flatten().reshape(1, -1)

# grayscale only for quantum
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Image processed")

# ===============================
# AI PREDICTION (STABLE)
# ===============================

predictions = []

for _ in range(5):
    pred = model.predict(features)[0]
    predictions.append(pred)

prediction = max(set(predictions), key=predictions.count)

plant, disease = prediction.split("___")

print("\nAI Prediction:", prediction)

# Confidence
confidence = np.max(model.predict_proba(features)) * 100
print(f"Confidence: {confidence:.2f}%")

# ===============================
# CREATE QUANTUM CIRCUIT
# ===============================

from qiskit import QuantumCircuit

qc = QuantumCircuit(2, 2)

mean_val = np.mean(gray) / 255.0

if mean_val > 0.5:
    qc.x(0)

qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

print("Quantum Circuit Created")

# ===============================
# CONNECT TO IBM QUANTUM
# ===============================

from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token="VoilVPZh83_0DhVdx8JdtrjqjrmowsR366tR-0tEb6ag"
)

backend = service.least_busy(simulator=False)

print("Using backend:", backend)

# ===============================
# TRANSPILING
# ===============================

from qiskit import transpile

qc_transpiled = transpile(qc, backend)

# ===============================
# RUN CIRCUIT
# ===============================

from qiskit_ibm_runtime import Sampler

sampler = Sampler(backend)

job = sampler.run([qc_transpiled], shots=1024)

print("Running quantum job...")

result = job.result()
counts = result[0].data.c.get_counts()

print("Quantum Output:", counts)

# ===============================
# HYBRID DECISION LOGIC
# ===============================

dominant_state = max(counts, key=counts.get)

print("Dominant Quantum State:", dominant_state)

print("\n==============================")
print("FINAL RESULT (HYBRID)")
print("==============================")

print("Plant:", plant)

# Improved final logic
if disease.lower() != "healthy" and dominant_state == "11":
    print(f"Disease Confirmed: {disease} ❌")
elif disease.lower() == "healthy" and dominant_state == "00":
    print("Status: Healthy Plant ✅")
else:
    print("Result Uncertain (Quantum Variation) ⚠️")
    print("AI suggests:", disease)

print("\nWorking ✅")