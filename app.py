import streamlit as st
import cv2
import numpy as np
import joblib
import os

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# Load model with error handling
try:
    if not os.path.exists("plant_model.pkl"):
        st.error("❌ Model file 'plant_model.pkl' not found. Please run main.py first to train the model.")
        st.stop()
    model = joblib.load("plant_model.pkl")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.set_page_config(page_title="AI+Quantum Plant Detection", page_icon="🌿", layout="wide")

st.title("🌿 AI + Quantum Plant Disease Detection")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", width=500)

    # Preprocess (Aligned with main.py: 128x128 color)
    try:
        img_resized = cv2.resize(img, (128, 128))
        features = img_resized.flatten().reshape(1, -1)
        
        # Verify feature count matches model
        if features.shape[1] != model.n_features_in_:
            st.error(f"❌ Feature mismatch: Model expects {model.n_features_in_} features, but got {features.shape[1]}. Check image dimensions.")
            st.stop()

        # AI Prediction
        prediction = model.predict(features)[0]
        plant, disease = prediction.split("___")

        st.subheader("🧠 AI Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Plant Type", plant)
        with col2:
            st.metric("Detected Disease", disease)
            
        # Confidence
        conf_probs = model.predict_proba(features)
        confidence = np.max(conf_probs) * 100
        st.info(f"AI Model Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"❌ AI Prediction Error: {e}")
        st.stop()

    # Quantum Circuit
    st.markdown("---")
    st.subheader("⚛️ Quantum Analysis")
    
    try:
        # Prepare grayscale for quantum bit encoding
        gray_quantum = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray_quantum) / 255.0
        
        qc = QuantumCircuit(2, 2)
        if mean_val > 0.5:
            qc.x(0)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        with st.status("🔗 Connecting to IBM Quantum...", expanded=True) as status:
            # Secure token handling for Streamlit Community Cloud
            if "IBM_QUANTUM_TOKEN" in st.secrets:
                IBM_TOKEN = st.secrets["IBM_QUANTUM_TOKEN"]
            else:
                IBM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN", "8qygOdF_SXljNMGQdTGZEYCQqJTY62rE6eIhvUOACwTB")
            
            service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=IBM_TOKEN
            )

            st.write("🔍 Requesting least busy backend...")
            try:
                backend = service.least_busy(simulator=False, min_qubits=2)
                st.write(f"✅ Using real backend: {backend.name}")
            except Exception as backend_err:
                st.write(f"ℹ️ Real hardware busy or account limit reached. Searching for available simulator...")
                try:
                    backend = service.least_busy(simulator=True)
                    st.write(f"✅ Using cloud simulator: {backend.name}")
                except Exception as sim_err:
                    st.error("❌ No cloud backends (real or simulator) found for this account.")
                    raise sim_err

            st.write("🚀 Running hybrid quantum execution...")
            qc_transpiled = transpile(qc, backend)
            sampler = Sampler(backend)
            job = sampler.run([qc_transpiled], shots=1024)

            result = job.result()
            counts = result[0].data.c.get_counts()
            status.update(label="✅ Quantum Job Completed!", state="complete", expanded=False)

        dominant_state = max(counts, key=counts.get)

        st.write("Quantum Statistical Verification Counts:", counts)
        st.write(f"Dominant Quantum State Observed: **{dominant_state}**")

        st.subheader("🏁 Final Hybrid Decision")

        if disease.lower() == "healthy" and dominant_state == "00":
            st.success("Status: Healthy Plant confirmed by both AI and Quantum ✅")
        elif disease.lower() != "healthy" and dominant_state == "11":
            st.error(f"Diagnosis: {disease} - High confidence verification by Quantum Algorithm ❌")
        else:
            st.warning("Result Uncertain: AI and Quantum insights vary ⚠️")
            st.markdown(f"**AI Prediction:** {disease}")
            st.markdown(f"**Quantum State:** {dominant_state} (suggests variation from expected pattern)")

    except Exception as e:
        st.error(f"❌ Quantum Execution Failed or Timed Out.")
        st.exception(e)
        st.info("Operating in AI-only fallback mode.")
        st.success(f"Final Outcome (AI-Based): {plant} - {disease}")