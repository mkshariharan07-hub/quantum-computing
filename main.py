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

try:
    if not os.path.exists("images.npy") or not os.path.exists("labels.npy"):
        raise FileNotFoundError("Missing dataset files (images.npy or labels.npy). Please ensure they are in the directory.")
    
    images = np.load("images.npy")
    labels = np.load("labels.npy")

    print(f"✅ Dataset loaded successfully. Total images: {len(images)}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

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

try:
    if not os.path.exists("plant_model.pkl"):
        print("\n⏳ Training model (this may take a moment)...")

        if len(processed_images) == 0:
            raise ValueError("No images found to train on.")

        X_train, X_test, y_train, y_test = train_test_split(
            processed_images, labels, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        joblib.dump(model, "plant_model.pkl")
        print("✅ Model trained and saved as plant_model.pkl")

    else:
        model = joblib.load("plant_model.pkl")
        print("✅ Model loaded successfully from plant_model.pkl")
        
        # Verify feature compatibility
        expected_features = 128 * 128 * 3
        if model.n_features_in_ != expected_features:
            print(f"⚠️ Model mismatch! Expected {model.n_features_in_} features, but script is set for {expected_features}.")
            print("Consider deleting plant_model.pkl to retrain.")

except Exception as e:
    print(f"❌ Error during model operations: {e}")
    exit(1)

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

try:
    predictions = []
    # Using predict_proba for better confidence estimation if needed
    for _ in range(5):
        pred = model.predict(features)[0]
        predictions.append(pred)

    prediction = max(set(predictions), key=predictions.count)
    plant, disease = prediction.split("___")

    print(f"\n🤖 AI Prediction: {prediction}")

    # Confidence calculation
    conf_probs = model.predict_proba(features)
    confidence = np.max(conf_probs) * 100
    print(f"📊 Confidence Score: {confidence:.2f}%")

except Exception as e:
    print(f"❌ Prediction Error: {e}")
    print("Ensure the input image matches the expected training format.")
    exit(1)

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

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
    from qiskit import transpile
    from dotenv import load_dotenv
    load_dotenv() # Load tokens from .env file if it exists

    print("\n🔗 Connecting to IBM Quantum...")
    IBM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN", "VoilVPZh83_0DhVdx8JdtrjqjrmowsR366tR-0tEb6ag")
    
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=IBM_TOKEN
    )

    # Use a simulator if no real backend is available or for testing speed
    try:
        backend = service.least_busy(simulator=False, min_qubits=2)
        print(f"✅ Using real backend: {backend.name}")
    except:
        backend = service.get_backend("ibmq_qasm_simulator")
        print("ℹ️ No real backend available, falling back to simulator.")

    # ===============================
    # RUN CIRCUIT
    # ===============================
    print("🚀 Transpiling and running quantum circuit...")
    qc_transpiled = transpile(qc, backend)
    sampler = Sampler(backend)
    job = sampler.run([qc_transpiled], shots=1024)

    # Wait for result with a timeout or error handling
    result = job.result()
    counts = result[0].data.c.get_counts()

    print("⚛️ Quantum Output Counts:", counts)

except ImportError:
    print("❌ Qiskit modules not found. Please install: pip install qiskit-ibm-runtime qiskit")
    exit(1)
except Exception as e:
    print(f"❌ Quantum Execution Error: {e}")
    print("Fallback: Using AI prediction only due to Quantum Backend failure.")
    counts = {"00": 512, "11": 512} # Neutral fallback

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