import streamlit as st
import cv2
import numpy as np
import joblib

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# Load model
model = joblib.load("plant_model.pkl")

st.title("🌿 AI + Quantum Plant Disease Detection")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", width=500)

    # Preprocess
    img_resized = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    features = gray.flatten().reshape(1, -1)

    # AI Prediction
    prediction = model.predict(features)[0]
    plant, disease = prediction.split("___")

    st.subheader("🧠 AI Prediction")
    st.write("Plant:", plant)
    st.write("Disease:", disease)

    # Quantum Circuit
    qc = QuantumCircuit(2, 2)

    mean_val = np.mean(gray) / 255.0
    if mean_val > 0.5:
        qc.x(0)

    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    st.subheader("⚛️ Running Quantum...")

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token="8qygOdF_SXljNMGQdTGZEYCQqJTY62rE6eIhvUOACwTB"
    )

    backend = service.least_busy(simulator=False)
    qc_transpiled = transpile(qc, backend)

    sampler = Sampler(backend)
    job = sampler.run([qc_transpiled], shots=1024)

    result = job.result()
    counts = result[0].data.c.get_counts()

    dominant_state = max(counts, key=counts.get)

    st.subheader("⚛️ Quantum Output")
    st.write(counts)
    st.write("Dominant State:", dominant_state)

    st.subheader("✅ Final Result")

    if disease.lower() == "healthy" and dominant_state == "00":
        st.success("Healthy Plant ✅")
    elif disease.lower() != "healthy" and dominant_state == "11":
        st.error(f"Disease Confirmed: {disease} ❌")
    else:
        st.warning("Mismatch⚠️")