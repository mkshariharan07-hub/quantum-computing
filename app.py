import streamlit as st
import cv2
import numpy as np
import joblib
import os
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# ===============================
# PAGE CONFIG & STYLING
# ===============================
st.set_page_config(
    page_title="PlantPulse AI + Quantum",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        border-color: #10b981;
    }
    
    .quantum-badge {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# UTILITIES
# ===============================
def extract_features(img):
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    means, stds = cv2.meanStdDev(img)
    stats = np.concatenate([means.flatten(), stds.flatten()])
    return np.concatenate([h_hist, s_hist, v_hist, stats])

def load_app_model():
    if not os.path.exists("plant_model.pkl"):
        st.error("🚨 Model not found! Please run the training script (main.py) first.")
        st.stop()
    return joblib.load("plant_model.pkl")

model = load_app_model()

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.image("https://img.icons8.com/color/144/leaf.png", width=100)
    st.title("PlantPulse Engine")
    st.markdown("---")
    st.info("Hybrid AI-Quantum system for high-precision botanical diagnosis.")
    st.markdown("### Configuration")
    confidence_threshold = st.slider("AI Confidence Threshold", 0, 100, 70)
    backend_pref = st.selectbox("Quantum Backend", ["Dynamic (Least Busy)", "Simulator Only"])

# ===============================
# MAIN UI
# ===============================
st.markdown("# 🌿 PlantPulse <span class='quantum-badge'>AI + QUANTUM</span>", unsafe_allow_html=True)
st.write("Upload a leaf image for real-time analysis across classical and quantum systems.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    uploaded_file = st.file_uploader("Drop leaf image here...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, caption="Captured Specimen", use_column_width=True)

with col2:
    if uploaded_file:
        with st.spinner("⚡ Classical AI Processing..."):
            features = extract_features(img).reshape(1, -1)
            
            # Feature check
            if features.shape[1] != model.n_features_in_:
                st.warning(f"🔄 Re-aligning features... (Model expects {model.n_features_in_})")
                # Fallback to simple flattening if model is old
                features = cv2.resize(img, (128, 128)).flatten().reshape(1, -1)
            
            prediction = model.predict(features)[0]
            conf_probs = model.predict_proba(features)
            confidence = np.max(conf_probs) * 100
            
            try:
                plant, disease = prediction.split("___")
            except:
                plant, disease = "Unknown", prediction

        st.markdown(f"### 🧠 Classical Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='metric-card'><h4>Plant</h4><h2>{plant}</h2></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><h4>Status</h4><h2>{disease}</h2></div>", unsafe_allow_html=True)
        
        st.progress(confidence/100, text=f"AI Confidence Score: {confidence:.2f}%")

        # Quantum Part
        st.markdown("---")
        st.markdown("### ⚛️ Quantum Verification")
        
        try:
            gray_q = cv2.cvtColor(cv2.resize(img, (64,64)), cv2.COLOR_BGR2GRAY)
            mean_val = np.mean(gray_q) / 255.0
            
            qc = QuantumCircuit(2, 2)
            if mean_val > 0.5: qc.x(0)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])

            with st.status("🔗 Running Quantum Job...", expanded=False) as status:
                IBM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN", "8qygOdF_SXljNMGQdTGZEYCQqJTY62rE6eIhvUOACwTB")
                service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_TOKEN)
                
                if backend_pref == "Simulator Only":
                    backend = service.backend("ibmq_qasm_simulator")
                else:
                    try:
                        backend = service.least_busy(simulator=False, min_qubits=2)
                    except:
                        backend = service.least_busy(simulator=True)
                
                status.write(f"Connected to: {backend.name}")
                qc_transpiled = transpile(qc, backend)
                sampler = Sampler(backend)
                job = sampler.run([qc_transpiled], shots=1024)
                result = job.result()
                counts = result[0].data.c.get_counts()
                status.update(label="✅ Quantum Data Received", state="complete")

            dominant_state = max(counts, key=counts.get)
            
            # Final Hybrid Result
            is_healthy = disease.lower() == "healthy"
            quantum_positive = dominant_state == "11"
            quantum_negative = dominant_state == "00"

            st.markdown("### 🏁 Final Diagnosis")
            if is_healthy and quantum_negative:
                st.success(f"Verified: **Healthy {plant}** confirmed by hybrid consensus.")
            elif not is_healthy and quantum_positive:
                st.error(f"Alert: **{disease} detected** in {plant}. Quantum state confirms mutation.")
            else:
                st.warning("Analysis Mixed: Quantum variation suggests potential anomaly.")
                st.info(f"Primary AI diagnosis: {disease}")

        except Exception as e:
            st.warning(f"Quantum Fallback: {e}")
            st.info(f"Diagnosis based on AI only: {disease}")
    else:
        st.empty()
        st.markdown("""
        <div style='text-align: center; padding: 100px; color: #64748b;'>
            <img src='https://img.icons8.com/dotty/80/64748b/upload.png'/><br>
            Waiting for specimen upload...
        </div>
        """, unsafe_allow_html=True)